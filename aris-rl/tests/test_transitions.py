"""Tests for replay transition dict helpers."""

from __future__ import annotations

import numpy as np

from aris_rl.training.transitions import row_dict_to_numpy


def test_row_dict_to_numpy_shapes() -> None:
    """Replay import uses fixed shapes so storage stays compatible with training."""
    row = {
        "obs": [0.1] * 12,
        "next_obs": [0.2] * 12,
        "action": [3],
        "reward": 1.5,
        "done": False,
    }
    obs, next_obs, action, reward, done = row_dict_to_numpy(
        row,
        state_dim=12,
        action_shape=(1,),
        action_dtype=np.dtype("int32"),
    )
    assert obs.shape == (12,)
    assert next_obs.shape == (12,)
    assert action.shape == (1,)
    assert reward == 1.5
    assert done is False
