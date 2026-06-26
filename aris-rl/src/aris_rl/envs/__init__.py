"""Gymnasium environments and helpers for actions and simulation backends."""

from aris_rl.envs.backends import MockBackend, SimulationBackend, StepOutcome
from aris_rl.envs.continuous_actions import decode_continuous_action, make_ppo_box_space
from aris_rl.envs.discrete_actions import (
    DiscreteActionLayout,
    decode_discrete_action,
    discrete_layout_from_config,
    encode_discrete_action,
    make_multi_discrete_space,
)
from aris_rl.envs.flat_actions import flat_dim, flat_to_multi, multi_to_flat
from aris_rl.envs.wrappers import FlattenMultiDiscreteActions

__all__ = [
    "DiscreteActionLayout",
    "FlattenMultiDiscreteActions",
    "MockBackend",
    "SimulationBackend",
    "StepOutcome",
    "decode_continuous_action",
    "decode_discrete_action",
    "discrete_layout_from_config",
    "encode_discrete_action",
    "flat_dim",
    "flat_to_multi",
    "make_multi_discrete_space",
    "make_ppo_box_space",
    "multi_to_flat",
]
