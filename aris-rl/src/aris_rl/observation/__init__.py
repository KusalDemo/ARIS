"""
Turn raw metrics and policy settings into the fixed 12-number state for the learner.

Import from here, e.g. `from aris_rl.observation import StateAggregator`.
"""

from aris_rl.observation.aggregator import (
    PolicyContext,
    StateAggregator,
    aggregator_from_config,
    build_state_vector,
)
from aris_rl.observation.features import FEATURE_NAMES, STATE_DIM

__all__ = [
    "FEATURE_NAMES",
    "STATE_DIM",
    "PolicyContext",
    "StateAggregator",
    "aggregator_from_config",
    "build_state_vector",
]
