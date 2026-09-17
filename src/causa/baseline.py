"""Chance baselines.

A raw ownership score means nothing on its own. A system whose predictions
are pure noise still scores well above zero — around 0.40 under the default
config, which has a floor of 0.25 and a surprise floor of 0.38 — and a
system whose state space is small and tightly clustered can score high for
purely geometric reasons.

The only honest way to report a score is against the score the same data
produces when the pairing between intention and outcome is destroyed. If
the real score does not clear the shuffled one, the measured "ownership"
is an artefact of the state space, not evidence of agency.

This module is the reason the library exists as a library rather than as a
single function.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .core import DirectionalConfig, Vector, directional_ownership

__all__ = ["BaselineReport", "Episode", "permutation_baseline"]


@dataclass(frozen=True)
class Episode:
    """One intention-outcome pair."""

    before: Vector
    predicted: Vector
    after: Vector
    label: str = ""


@dataclass(frozen=True)
class BaselineReport:
    observed_mean: float
    baseline_mean: float
    baseline_stdev: float
    effect: float
    """``observed_mean - baseline_mean``. Positive means outcomes tracked
    intentions better than chance pairing would produce."""

    p_value: float
    """Fraction of shuffles whose mean matched or exceeded the observed
    mean. One-sided, permutation-based, no distributional assumptions."""

    n_episodes: int

    n_defined: int
    """Episodes that permitted a real comparison in the *observed* pairing.
    Shuffled pairings can yield a different count, since whether an episode
    is defined depends on the outcome it is paired with. When the two
    counts diverge sharply the means are taken over different subsets and
    the comparison weakens — check ``n_defined`` against ``n_episodes``
    before reading much into a small effect."""

    n_permutations: int

    def summary(self) -> str:
        return (
            f"observed={self.observed_mean:.3f} "
            f"baseline={self.baseline_mean:.3f}±{self.baseline_stdev:.3f} "
            f"effect={self.effect:+.3f} p={self.p_value:.4f} "
            f"(n={self.n_defined}/{self.n_episodes} defined, "
            f"{self.n_permutations} permutations)"
        )


def permutation_baseline(
    episodes: Sequence[Episode],
    n_permutations: int = 1000,
    config: DirectionalConfig | None = None,
    seed: int | None = None,
    include_undefined: bool = False,
) -> BaselineReport:
    """Compare observed ownership against shuffled intention-outcome pairing.

    Each permutation keeps every ``(before, predicted)`` intention and every
    ``after`` outcome, but reassigns which outcome followed which intention.
    Everything about the state space — its scale, its clustering, its
    dimensionality — is preserved. Only the causal pairing is destroyed.

    Parameters
    ----------
    include_undefined:
        Whether to include episodes where no real comparison was possible
        (no movement, or no intended direction). Default is to exclude
        them, since their score is a stated prior rather than a
        measurement, and including them pulls both means toward that prior.
    """
    if len(episodes) < 3:
        raise ValueError("need at least 3 episodes for a permutation baseline")

    rng = random.Random(seed)
    cfg = config or DirectionalConfig()

    def mean_over(pairings: Sequence[int]) -> tuple[float, int]:
        scores = []
        for i, j in enumerate(pairings):
            r = directional_ownership(
                episodes[i].before, episodes[i].predicted, episodes[j].after, cfg
            )
            if r.defined or include_undefined:
                scores.append(r.score)
        if not scores:
            return float("nan"), 0
        return statistics.fmean(scores), len(scores)

    identity = list(range(len(episodes)))
    observed_mean, n_defined = mean_over(identity)

    if n_defined == 0:
        raise ValueError(
            "no episode permitted a real comparison; "
            "check that predictions differ from starting states"
        )

    shuffled_means = []
    at_or_above = 0
    for _ in range(n_permutations):
        perm = identity[:]
        rng.shuffle(perm)
        m, _ = mean_over(perm)
        if math.isnan(m):
            continue
        shuffled_means.append(m)
        if m >= observed_mean:
            at_or_above += 1

    if not shuffled_means:
        raise ValueError("every permutation was undefined; baseline unusable")

    baseline_mean = statistics.fmean(shuffled_means)
    baseline_stdev = (
        statistics.stdev(shuffled_means) if len(shuffled_means) > 1 else 0.0
    )

    return BaselineReport(
        observed_mean=observed_mean,
        baseline_mean=baseline_mean,
        baseline_stdev=baseline_stdev,
        effect=observed_mean - baseline_mean,
        p_value=(at_or_above + 1) / (len(shuffled_means) + 1),
        n_episodes=len(episodes),
        n_defined=n_defined,
        n_permutations=len(shuffled_means),
    )
