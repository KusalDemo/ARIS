"""
Gymnasium wrapper: outer action is one integer, inner env still uses MultiDiscrete.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aris_rl.envs.flat_actions import flat_dim, flat_to_multi, nvec_from_space


class FlattenMultiDiscreteActions(gym.Wrapper):
    """DQN + SB3: Discrete outside, MultiDiscrete inside."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            msg = "FlattenMultiDiscreteActions needs a MultiDiscrete inner space"
            raise TypeError(msg)
        self._nvec = nvec_from_space(env.action_space)
        n = flat_dim(self._nvec)
        self.action_space = spaces.Discrete(n)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        inner = flat_to_multi(self._nvec, int(action))
        o, r, term, trunc, info = self.env.step(inner)
        return o, float(r), term, trunc, info

    def action(self, action: Any) -> np.ndarray:
        return flat_to_multi(self._nvec, int(action))
