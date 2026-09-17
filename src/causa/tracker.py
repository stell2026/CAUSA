"""Stateful tracking across a run.

The functions in :mod:`causa.core` are stateless by design. Most host
systems want a running picture instead: a bounded history, a smoothed
confidence, and a log of the moments where ownership departed sharply from
the middle. That is what this provides, and nothing more.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from .core import DirectionalConfig, OwnershipResult, Vector, directional_ownership

__all__ = ["OwnershipEvent", "OwnershipTracker"]


@dataclass(frozen=True)
class OwnershipEvent:
    step: int
    label: str
    score: float
    defined: bool


class OwnershipTracker:
    """Track directional ownership over a sequence of intentions.

    Usage is two-phase per step, mirroring how intention and outcome are
    separated in time::

        tracker.register_intent("answer honestly", state_now, predicted)
        ...                                  # the action happens
        result = tracker.observe(state_now)  # scores the previous intent

    Calling :meth:`observe` with no registered intention returns ``None``
    rather than inventing a score.

    An event is logged when a score departs from ``event_reference`` by
    more than ``event_threshold``. The reference defaults to the midpoint
    of the configured score range rather than to 0.5, because the range is
    not centred on 0.5: with the default ``floor=0.25`` no score can ever
    fall below 0.2, so a reference of 0.5 would make the low side of the
    log unreachable and the event log silently one-sided. Pass
    ``event_reference=0.5`` to restore the original behaviour.

    Even with the midpoint reference the low side is narrow under the
    default config, because ``surprise_floor`` holds any substantial
    movement at or above 0.38: what still registers is a small movement
    that went the wrong way. Lower ``floor`` and ``surprise_floor`` if you
    want the low half of the log to carry as much as the high half.
    """

    def __init__(
        self,
        history_size: int = 30,
        event_size: int = 20,
        event_threshold: float = 0.3,
        event_reference: float | None = None,
        confidence_alpha: float = 0.15,
        initial_confidence: float = 0.5,
        config: DirectionalConfig | None = None,
    ) -> None:
        self.config = config or DirectionalConfig()
        self.event_threshold = event_threshold
        self.event_reference = (
            event_reference
            if event_reference is not None
            else (self.config.floor + self.config.ceiling) / 2.0
        )
        self.confidence_alpha = confidence_alpha
        self.confidence = initial_confidence
        self.step = 0

        self._intent: tuple[str, list[float], list[float]] | None = None
        self.history: deque[float] = deque(maxlen=history_size)
        self.events: deque[OwnershipEvent] = deque(maxlen=event_size)

    @property
    def pending_intent(self) -> str | None:
        return None if self._intent is None else self._intent[0]

    def register_intent(
        self, label: str, state_now: Vector, state_predicted: Vector
    ) -> None:
        """Record an intention and the state change it predicts."""
        self._intent = (label, list(state_now), list(state_predicted))

    def observe(self, state_after: Vector) -> OwnershipResult | None:
        """Score the pending intention against the observed state.

        Returns ``None`` when no intention is pending. Consumes the
        intention either way, so a stale intention never scores a later
        outcome.
        """
        if self._intent is None:
            return None

        label, before, predicted = self._intent
        self._intent = None
        self.step += 1

        result = directional_ownership(before, predicted, state_after, self.config)

        self.history.append(result.score)
        self.confidence = (
            self.confidence * (1.0 - self.confidence_alpha)
            + result.score * self.confidence_alpha
        )

        if abs(result.score - self.event_reference) > self.event_threshold:
            self.events.append(
                OwnershipEvent(
                    step=self.step,
                    label=label,
                    score=round(result.score, 3),
                    defined=result.defined,
                )
            )

        return result

    def discard_intent(self) -> None:
        """Drop a pending intention without scoring it."""
        self._intent = None

    def mean(self) -> float | None:
        """Mean ownership over the retained history, or ``None`` if empty."""
        if not self.history:
            return None
        return sum(self.history) / len(self.history)

    def trend(self, window: int = 10) -> float | None:
        """Difference between the last ``window`` scores and the ones before.

        Positive means ownership has been rising. Returns ``None`` when
        there is not enough history for both halves — which is permanent if
        ``window * 2`` exceeds ``history_size``.
        """
        if len(self.history) < window * 2:
            return None
        h = list(self.history)
        recent = h[-window:]
        earlier = h[-window * 2 : -window]
        return sum(recent) / window - sum(earlier) / window

    def __iter__(self) -> Iterator[float]:
        return iter(self.history)

    def __len__(self) -> int:
        return len(self.history)
