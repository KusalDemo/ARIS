"""
Turn raw metrics and policy settings into the fixed 12-number state for the learner.

For now we only expose the feature names and vector size. Aggregators and Prometheus
feeds are added in later commits.
"""

from aris_rl.observation.features import FEATURE_NAMES, STATE_DIM

__all__ = [
    "FEATURE_NAMES",
    "STATE_DIM",
]
