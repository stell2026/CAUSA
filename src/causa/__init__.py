"""CAUSA — Causal Agency and Utterance-State Alignment.

Two measurements of whether an artificial agent's outputs are causally
tied to its own internal state, plus the permutation baselines needed to
tell a real effect from an artefact of the state space.

See the README for scope and, more importantly, for what these metrics do
not measure.
"""

from .adapters import affective_projection, neuromodulator_projection
from .baseline import BaselineReport, Episode, permutation_baseline
from .core import (
    CongruenceConfig,
    CongruenceResult,
    DirectionalConfig,
    OwnershipResult,
    directional_ownership,
    expression_congruence,
)
from .tracker import OwnershipEvent, OwnershipTracker

__version__ = "0.1.0"

__all__ = [
    "BaselineReport",
    "CongruenceConfig",
    "CongruenceResult",
    "DirectionalConfig",
    "Episode",
    "OwnershipEvent",
    "OwnershipResult",
    "OwnershipTracker",
    "__version__",
    "affective_projection",
    "directional_ownership",
    "expression_congruence",
    "neuromodulator_projection",
    "permutation_baseline",
]
