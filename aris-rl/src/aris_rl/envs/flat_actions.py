"""
Stable-Baselines3 DQN wants one Discrete action, not MultiDiscrete.

We pack three small indices into one integer (like row-major coordinates in a 3-D grid)
and unpack before stepping the real environment.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import MultiDiscrete


def flat_dim(nvec: np.ndarray | tuple[int, ...]) -> int:
    v = np.asarray(nvec, dtype=np.int64)
    return int(np.prod(v))


def multi_to_flat(nvec: np.ndarray | tuple[int, ...], components: np.ndarray) -> int:
    v = np.asarray(nvec, dtype=np.int64)
    c = np.asarray(components, dtype=np.int64).reshape(-1)
    if c.shape[0] != v.shape[0]:
        msg = "component count must match nvec length"
        raise ValueError(msg)
    idx = 0
    m = 1
    for i in range(len(v) - 1, -1, -1):
        if not (0 <= c[i] < v[i]):
            msg = f"component {i}={c[i]} out of range for nvec[i]={v[i]}"
            raise ValueError(msg)
        idx += int(c[i]) * m
        m *= int(v[i])
    return idx


def flat_to_multi(nvec: np.ndarray | tuple[int, ...], flat: int) -> np.ndarray:
    v = np.asarray(nvec, dtype=np.int64)
    k = int(flat)
    out = np.zeros(len(v), dtype=np.int64)
    for i in range(len(v) - 1, -1, -1):
        out[i] = k % int(v[i])
        k //= int(v[i])
    return out.astype(np.int32, copy=False)


def nvec_from_space(action_space: MultiDiscrete) -> np.ndarray:
    return np.asarray(action_space.nvec, dtype=np.int64)
