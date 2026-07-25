"""
Stable-Baselines3 hook: after each simulator step, bump Prometheus counters and reward EMA.

A "callback" here is just a plug-in SB3 calls while learning — we use it to mirror replay
health on the same /metrics page operators might already watch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """Feeds step rewards and action counts to :class:`TrainingMetricsExporter`."""

    def __init__(self, exporter: Any, algorithm: str) -> None:
        super().__init__()
        self._exporter = exporter
        self._algorithm = algorithm.lower()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is None:
            return True
        r = float(np.mean(np.asarray(rewards, dtype=np.float64)))
        self._exporter.record_env_step(algorithm=self._algorithm, reward=r)
        return True
