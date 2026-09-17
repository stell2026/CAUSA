"""Reference projections into the congruence space.

:func:`causa.core.expression_congruence` deliberately takes abstract
dimension maps. Something has to produce them, and that something is
architecture-specific. This module holds one worked example — the
projection used by the system these metrics were extracted from — so that
the mapping is documented rather than folded silently into the metric.

Treat it as a template to copy and adapt, not as a general solution.

Both projections land in the same box: ``valence`` and ``arousal``, each in
``[-1, 1]``. That is a range of 2, so compare them with
``CongruenceConfig(scale=2.0)`` — the default scale of 1.0 assumes a
dimension in ``[0, 1]`` and drives every mismatch above 1.0 to zero.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["affective_projection", "neuromodulator_projection"]


def _clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def affective_projection(
    stimulus: Mapping[str, float],
    tension_weight: float = 0.5,
) -> dict[str, float]:
    """Project a stimulus decomposition into ``valence`` and ``arousal``.

    ``stimulus`` is expected to carry some of the keys ``satisfaction``,
    ``tension`` and ``arousal``, each roughly in ``[0, 1]``. In the source
    system this decomposition is produced from the text of an utterance,
    which is why it is the *expressed* side of the comparison.

    Missing keys are treated as zero — a stimulus that says nothing about
    tension is not the same as one that asserts low tension, but the
    difference cannot be recovered here and is flattened deliberately
    rather than guessed at.

    Both outputs are clamped to ``[-1, 1]``, which matters because arousal
    is a sum: without the clamp it reaches 1.5 while
    :func:`neuromodulator_projection` stops at 1.0, and the two would no
    longer live in the same space.
    """
    satisfaction = float(stimulus.get("satisfaction", 0.0))
    tension = float(stimulus.get("tension", 0.0))
    arousal = float(stimulus.get("arousal", 0.0))

    return {
        "valence": _clamp_unit(satisfaction - tension),
        "arousal": _clamp_unit(arousal + tension * tension_weight),
    }


def neuromodulator_projection(
    levels: Mapping[str, float],
    serotonin_weight: float = 0.6,
    dopamine_weight: float = 0.4,
) -> dict[str, float]:
    """Project simulated neuromodulator levels into ``valence``/``arousal``.

    ``levels`` is expected to carry ``serotonin``, ``dopamine`` and
    ``noradrenaline``, each in ``[0, 1]`` with 0.5 as the neutral point.
    Valence is a weighted deviation of the first two from neutral; arousal
    is noradrenaline rescaled to ``[-1, 1]``.

    The weights encode a claim about the modelled system — that tonic mood
    contributes more to valence than reward signalling does. That claim is
    inherited from the source architecture and is not defended here; change
    the weights if your system says otherwise.
    """
    serotonin = float(levels.get("serotonin", 0.5))
    dopamine = float(levels.get("dopamine", 0.5))
    noradrenaline = float(levels.get("noradrenaline", 0.5))

    valence = (serotonin - 0.5) * serotonin_weight + (dopamine - 0.5) * dopamine_weight
    arousal = (noradrenaline - 0.5) * 2.0

    return {"valence": _clamp_unit(valence), "arousal": _clamp_unit(arousal)}
