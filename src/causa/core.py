"""Core ownership metrics.

Two independent measurements, both pure functions over numeric state vectors:

``directional_ownership``
    Did the system's state move in the direction it intended?

``expression_congruence``
    Does what the system expressed match what its internal state was?

Neither requires a language model, a specific architecture, or a specific
state space. They require only that the host system can expose numeric
state vectors before and after an action, and (for the second metric) a
numeric encoding of what it expressed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Vector = Sequence[float]

__all__ = [
    "CongruenceConfig",
    "CongruenceResult",
    "DirectionalConfig",
    "OwnershipResult",
    "directional_ownership",
    "expression_congruence",
]


def _check(v: Vector, name: str) -> None:
    """Reject inputs that cannot produce a measurement.

    An empty vector or a non-finite component would otherwise flow through
    the arithmetic and come out as a plausible-looking number. A metric
    that silently invents a score for garbage input is worse than one that
    refuses.
    """
    if len(v) == 0:
        raise ValueError(
            f"{name} is empty; a state vector needs at least one dimension"
        )
    for i, x in enumerate(v):
        if not math.isfinite(x):
            raise ValueError(f"{name}[{i}] is not finite: {x!r}")


def _norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def _sub(a: Vector, b: Vector) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return [x - y for x, y in zip(a, b, strict=True)]


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _clamp(x: float, lo: float, hi: float) -> float:
    if math.isnan(x):
        return (lo + hi) / 2.0
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class DirectionalConfig:
    """Tunable constants for :func:`directional_ownership`.

    The defaults reproduce the behaviour of the original implementation.
    They are exposed because every one of them is a modelling choice, not a
    derived quantity — report the config alongside any published score.
    """

    floor: float = 0.25
    """Lower bound on the score. A system that acted at all is never fully
    non-causal. Note that this floor applies to every path through the
    function, including the ones where no real comparison was possible —
    those are flagged by ``defined=False``, not by a distinguished score.
    Set ``floor=0.0`` if you want the full range."""

    ceiling: float = 1.0

    still_threshold: float = 0.01
    """Below this movement the outcome is treated as "nothing happened"."""

    still_ceiling: float = 0.75
    """When nothing happened, the score cannot claim strong ownership."""

    still_scale: float = 0.6
    still_offset: float = 0.25
    """When nothing happened, the score is
    ``directional * still_scale + still_offset``, bounded by ``floor`` and
    ``still_ceiling``. With the defaults that is ``[0.25, 0.85]`` before
    the ceiling clips it — a stated prior, not a measurement."""

    min_intent_magnitude: float = 0.01
    min_delta_magnitude: float = 0.005
    """Below these, direction is undefined and the score falls back to
    ``undefined_score`` rather than to noise."""

    undefined_score: float = 0.5

    movement_saturation: float = 3.0
    """How fast weight shifts from directional agreement to distance
    agreement as actual movement grows. Weight saturates at
    ``1 / movement_saturation`` units of movement."""

    surprise_threshold: float = 0.15
    surprise_floor: float = 0.38
    """If the state moved substantially, the system did something — even if
    the direction disagreed with the prediction. This prevents an external
    surprise from being scored as total non-agency."""

    epsilon: float = 0.01


@dataclass(frozen=True)
class CongruenceConfig:
    """Tunable constants for :func:`expression_congruence`."""

    weights: Mapping[str, float] | None = None
    """Per-dimension weights, keyed by dimension name. When ``None``, all
    dimensions are weighted equally."""

    scale: float = 1.0
    """Expected magnitude range of a single dimension — the largest
    mismatch that should still score above zero. Mismatch is divided by
    this before being subtracted from 1.0.

    The default suits dimensions in ``[0, 1]``. For a signed dimension in
    ``[-1, 1]`` the range is 2, and leaving this at 1.0 collapses every
    mismatch above 1.0 to a score of zero. The projections in
    :mod:`causa.adapters` produce signed dimensions, so pass
    ``CongruenceConfig(scale=2.0)`` when comparing them."""


@dataclass(frozen=True)
class OwnershipResult:
    score: float
    """Ownership in ``[floor, ceiling]``. Higher = the outcome tracked the
    intention more closely."""

    directional_component: float
    """Cosine agreement between intended direction and actual direction,
    rescaled to ``[0, 1]``. 0.5 means orthogonal or undefined."""

    distance_component: float
    """How close the outcome landed to the prediction, relative to how far
    it travelled from the starting point."""

    movement_weight: float
    """How much the final score leaned on distance rather than direction."""

    distance_to_predicted: float
    distance_travelled: float
    defined: bool
    """False when the inputs did not permit a real comparison (no movement,
    or no intended direction). The score is then a stated prior, not a
    measurement — exclude these from aggregate statistics."""

    def __float__(self) -> float:
        return self.score


@dataclass(frozen=True)
class CongruenceResult:
    score: float
    """Congruence in ``[0, 1]``. 1.0 = expressed state matches internal
    state exactly on every weighted dimension."""

    mismatch: float
    per_dimension: dict[str, float] = field(default_factory=dict)
    """Absolute mismatch per dimension, before weighting. Use this to see
    *where* expression and state diverged, not only that they did."""

    def __float__(self) -> float:
        return self.score


def directional_ownership(
    state_before: Vector,
    state_predicted: Vector,
    state_after: Vector,
    config: DirectionalConfig | None = None,
) -> OwnershipResult:
    """Score how far an outcome tracked an intention.

    The three vectors must share a dimensionality and a coordinate system.
    In the reference system they are affective states (valence, arousal,
    dominance), but nothing here assumes that.

    Parameters
    ----------
    state_before:
        Internal state at the moment the intention was formed.
    state_predicted:
        The state the system predicted it would be in afterwards. This is
        the operational definition of "intention" — an intention that makes
        no prediction cannot be scored.
    state_after:
        Internal state actually observed afterwards.

    Notes
    -----
    The score blends two signals:

    * **direction** — whether movement went the intended way, regardless of
      how far it got;
    * **distance** — whether it landed where predicted, relative to how far
      it travelled.

    Direction dominates for small movements, where distance ratios are
    dominated by noise; distance dominates for large ones, where direction
    alone is too permissive. The crossover is set by
    ``config.movement_saturation``.
    """
    cfg = config or DirectionalConfig()

    _check(state_before, "state_before")
    _check(state_predicted, "state_predicted")
    _check(state_after, "state_after")

    delta = _sub(state_after, state_before)
    intent_dir = _sub(state_predicted, state_before)

    distance_to_predicted = _norm(_sub(state_after, state_predicted))
    distance_travelled = _norm(delta)
    intent_mag = _norm(intent_dir)

    if (
        intent_mag > cfg.min_intent_magnitude
        and distance_travelled > cfg.min_delta_magnitude
    ):
        cos_sim = _dot(delta, intent_dir) / (distance_travelled * intent_mag)
        directional = _clamp((cos_sim + 1.0) / 2.0, 0.0, 1.0)
        direction_defined = True
    else:
        directional = cfg.undefined_score
        direction_defined = False

    if distance_travelled < cfg.still_threshold:
        score = _clamp(
            directional * cfg.still_scale + cfg.still_offset,
            cfg.floor,
            cfg.still_ceiling,
        )
        return OwnershipResult(
            score=score,
            directional_component=directional,
            distance_component=cfg.undefined_score,
            movement_weight=0.0,
            distance_to_predicted=distance_to_predicted,
            distance_travelled=distance_travelled,
            defined=False,
        )

    distance_component = _clamp(
        1.0 - distance_to_predicted / (distance_travelled + cfg.epsilon),
        cfg.floor,
        cfg.ceiling,
    )
    movement_weight = _clamp(distance_travelled * cfg.movement_saturation, 0.0, 1.0)

    score = movement_weight * distance_component + (1.0 - movement_weight) * directional

    if distance_travelled > cfg.surprise_threshold:
        score = max(score, cfg.surprise_floor)

    score = _clamp(score, cfg.floor, cfg.ceiling)

    return OwnershipResult(
        score=score,
        directional_component=directional,
        distance_component=distance_component,
        movement_weight=movement_weight,
        distance_to_predicted=distance_to_predicted,
        distance_travelled=distance_travelled,
        defined=direction_defined,
    )


def expression_congruence(
    expressed: Mapping[str, float],
    internal: Mapping[str, float],
    config: CongruenceConfig | None = None,
) -> CongruenceResult:
    """Score how far an expression matched the state it was expressed from.

    Both arguments map dimension names to values in the *same* space. How
    an utterance is projected into that space is the host system's problem
    and deliberately out of scope here — see :mod:`causa.adapters` for the
    reference projection used by the original system.

    A high score does not mean the expression was *true*. It means the
    expression and the measured internal state moved together. Whether the
    internal state itself is meaningful is a property of the host system,
    not of this metric.
    """
    cfg = config or CongruenceConfig()

    shared = set(expressed) & set(internal)
    if not shared:
        raise ValueError("expressed and internal share no dimensions")

    for k in sorted(shared):
        for side, mapping in (("expressed", expressed), ("internal", internal)):
            if not math.isfinite(mapping[k]):
                raise ValueError(f"{side}[{k!r}] is not finite: {mapping[k]!r}")

    per_dimension = {k: abs(expressed[k] - internal[k]) for k in sorted(shared)}

    if cfg.weights is None:
        weights = {k: 1.0 for k in shared}
    else:
        weights = {k: float(cfg.weights.get(k, 0.0)) for k in shared}

    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("weights sum to zero over the shared dimensions")

    mismatch = sum(per_dimension[k] * weights[k] for k in shared) / total_weight
    score = _clamp(1.0 - mismatch / cfg.scale, 0.0, 1.0)

    return CongruenceResult(
        score=score,
        mismatch=mismatch,
        per_dimension=per_dimension,
    )
