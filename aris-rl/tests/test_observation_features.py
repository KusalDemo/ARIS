"""Tests for the 12-number state vector: size, names, and stable ordering."""

from __future__ import annotations

import re

import aris_rl
from aris_rl.observation import FEATURE_NAMES, STATE_DIM
from aris_rl.observation import features as features_mod

# Same order as configs/aris.default.yaml observation.feature_order (added Week 1 Wed).
EXPECTED_FEATURE_ORDER: tuple[str, ...] = (
    "cpu_util",
    "memory_util",
    "queue_depth",
    "latency_p50_ms",
    "latency_p99_ms",
    "error_rate",
    "retry_count",
    "backoff_multiplier",
    "timeout_ms",
    "global_rps",
    "reserved_0",
    "reserved_1",
)


def test_state_dim_is_twelve() -> None:
    assert STATE_DIM == 12
    assert len(FEATURE_NAMES) == STATE_DIM


def test_feature_names_unique_snake_case() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    for name in FEATURE_NAMES:
        assert pattern.match(name), f"unexpected feature name: {name!r}"


def test_feature_order_matches_spec() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURE_ORDER


def test_package_reexports_match_features_module() -> None:
    assert aris_rl.FEATURE_NAMES == features_mod.FEATURE_NAMES
    assert aris_rl.STATE_DIM == features_mod.STATE_DIM


def test_public_all_lists_exports() -> None:
    assert "FEATURE_NAMES" in aris_rl.__all__
    assert "STATE_DIM" in aris_rl.__all__
    assert "__version__" in aris_rl.__all__
