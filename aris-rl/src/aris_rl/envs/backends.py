"""
Fake microservice cluster for one Gymnasium step.

The real system is too heavy for training loops, so a backend returns latency, success,
and load numbers after we pick retry/backoff/timeout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepOutcome:
    """What happened in one simulated timestep — feeds the state builder and reward."""

    latency_ms: float
    success: bool
    cpu_util: float
    memory_util: float
    queue_depth: float
    global_rps: float
    overload_penalty: float


class SimulationBackend(ABC):
    """Contract: reset episode state, then step once per agent decision."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal counters before a new training episode."""

    @abstractmethod
    def step(
        self,
        t: float,
        *,
        retry: int,
        backoff_multiplier: float,
        timeout_ms: float,
    ) -> StepOutcome:
        """Simulate traffic at simulation time t with the chosen policy knobs."""


class MockBackend(SimulationBackend):
    """
    Simple stand-in: latency rises with internal 'stress' and with aggressive retries.

    Same random seed → same jitter sequence (tests stay repeatable).
    """

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._stress = 0.25

    def reset(self) -> None:
        self._stress = 0.25

    def step(
        self,
        t: float,
        *,
        retry: int,
        backoff_multiplier: float,
        timeout_ms: float,
    ) -> StepOutcome:
        _ = t
        self._stress = float(np.clip(self._stress + float(self._rng.normal(0.0, 0.025)), 0.0, 1.0))
        base_lat = (
            35.0
            + 30.0 * self._stress
            + 10.0 * float(retry)
            + 6.0 * max(0.0, float(backoff_multiplier) - 1.0)
        )
        jitter = float(self._rng.normal(0.0, 6.0))
        latency_ms = max(1.0, base_lat + jitter)
        success = latency_ms <= float(timeout_ms) * 0.92

        cpu_util = float(np.clip(0.28 + 0.55 * self._stress + 0.04 * float(retry), 0.0, 1.0))
        memory_util = float(np.clip(0.22 + 0.5 * self._stress, 0.0, 1.0))
        queue_depth = max(
            0.0,
            40.0 * self._stress + 6.0 * float(retry) - 8.0 * float(backoff_multiplier),
        )
        global_rps = float(max(50.0, 180.0 + 900.0 * self._stress + self._rng.normal(0.0, 25.0)))
        overload_penalty = float(
            np.clip(cpu_util * 0.55 + min(1.0, queue_depth / 400.0) * 0.45, 0.0, 1.0)
        )

        if success:
            self._stress = max(0.0, self._stress - 0.04)
        else:
            self._stress = min(1.0, self._stress + 0.07)

        return StepOutcome(
            latency_ms=latency_ms,
            success=success,
            cpu_util=cpu_util,
            memory_util=memory_util,
            queue_depth=queue_depth,
            global_rps=global_rps,
            overload_penalty=overload_penalty,
        )
