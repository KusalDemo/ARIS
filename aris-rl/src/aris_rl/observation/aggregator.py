"""
Build the 12-number state vector from recent metrics and current retry/timeout settings.

You push samples over time (latency, errors, CPU, …), then call build_vector when the
agent needs a decision. Same history + same time → same vector (no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aris_rl.observation.features import STATE_DIM
from aris_rl.observation.normalization import (
    min_max_unit,
    percentile_linear,
    z_score_to_unit,
)
from aris_rl.observation.windows import TimedFloatBuffer, TimedRequestBuffer


@dataclass
class PolicyContext:
    """What the proxy is doing right now: retries, backoff strength, timeout."""

    retry_count: int = 0
    backoff_multiplier: float = 1.0
    timeout_ms: float = 2000.0


class StateAggregator:
    """
    Collects telemetry and policy knobs, then outputs one numpy vector of length STATE_DIM.

    Index meanings match FEATURE_NAMES in features.py.
    """

    def __init__(self, observation_cfg: dict[str, Any], policy_cfg: dict[str, Any]) -> None:
        self._obs = observation_cfg
        self._policy = policy_cfg
        self._norm = observation_cfg["normalization"]
        self._missing = str(observation_cfg["missing_value_strategy"])
        lat_w = float(observation_cfg["latency_window_s"])
        err_w = float(observation_cfg["error_window_s"])
        self._latency_buf = TimedFloatBuffer(window_s=lat_w)
        self._cpu_buf = TimedFloatBuffer(window_s=lat_w)
        self._mem_buf = TimedFloatBuffer(window_s=lat_w)
        self._queue_buf = TimedFloatBuffer(window_s=lat_w)
        self._rps_buf = TimedFloatBuffer(window_s=lat_w)
        self._request_buf = TimedRequestBuffer(window_s=err_w)
        self.policy_context = PolicyContext()
        self._clip_lo = float(self._norm["clip_min"])
        self._clip_hi = float(self._norm["clip_max"])
        self._mode = str(self._norm["mode"])
        self._lat_upper = float(self._norm["latency_ms_upper"])
        self._queue_cap = float(self._norm["queue_depth_cap"])
        self._ref_rps = float(observation_cfg["reference_rps"])
        self._z = self._norm.get("z_score") or {}

    def set_policy_context(
        self,
        *,
        retry_count: int | None = None,
        backoff_multiplier: float | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        """Update slots 6–8 in the vector (retry_count, backoff_multiplier, timeout_ms)."""
        if retry_count is not None:
            self.policy_context.retry_count = int(retry_count)
        if backoff_multiplier is not None:
            self.policy_context.backoff_multiplier = float(backoff_multiplier)
        if timeout_ms is not None:
            self.policy_context.timeout_ms = float(timeout_ms)

    def record_latency_ms(self, t: float, latency_ms: float) -> None:
        """Feeds latency_p50_ms and latency_p99_ms after we take percentiles."""
        self._latency_buf.append(t, latency_ms)

    def record_request(self, t: float, *, ok: bool) -> None:
        """Feeds error_rate: ok=False counts as a failure."""
        self._request_buf.record(t, not ok)

    def record_load(self, t: float, *, cpu: float, memory: float, queue_depth: float) -> None:
        """Feeds cpu_util, memory_util, queue_depth (CPU/memory as 0–1 fractions)."""
        self._cpu_buf.append(t, cpu)
        self._mem_buf.append(t, memory)
        self._queue_buf.append(t, queue_depth)

    def record_global_rps(self, t: float, rps: float) -> None:
        """Feeds global_rps — how hard the system is being hit."""
        self._rps_buf.append(t, rps)

    def raw_error_rate(self, now: float | None = None) -> float | None:
        """Real error fraction before normalization (used by reward / cascade logic later)."""
        t_now = self._default_now() if now is None else float(now)
        return self._request_buf.error_rate(t_now)

    def _norm_latency_pair(self, p50: float | None, p99: float | None) -> tuple[float, float]:
        def one(raw: float | None) -> float:
            if raw is None:
                return self._missing_scalar(pessimistic_high=True)
            if self._mode == "z_score":
                m = float(self._z.get("latency_ms_mean", 200.0))
                s = float(self._z.get("latency_ms_std", 150.0))
                return z_score_to_unit(raw, m, s, self._clip_lo, self._clip_hi)
            return min_max_unit(raw, 0.0, self._lat_upper, self._clip_lo, self._clip_hi)

        return one(p50), one(p99)

    def _norm_scalar_min_max(
        self,
        raw: float | None,
        lo: float,
        hi: float,
        *,
        pessimistic_high: bool,
    ) -> float:
        if raw is None:
            return self._missing_scalar(pessimistic_high=pessimistic_high)
        if self._mode == "z_score":
            return min_max_unit(raw, lo, hi, self._clip_lo, self._clip_hi)
        return min_max_unit(raw, lo, hi, self._clip_lo, self._clip_hi)

    def _missing_scalar(self, *, pessimistic_high: bool) -> float:
        # pessimistic: assume the worst (high) when data is missing; neutral: use 0.5
        if self._missing == "pessimistic":
            return self._clip_hi if pessimistic_high else self._clip_lo
        return (self._clip_lo + self._clip_hi) / 2.0

    def _norm_error_rate(self, rate: float | None) -> float:
        if rate is None:
            return self._missing_scalar(pessimistic_high=True)
        if self._mode == "z_score":
            m = float(self._z.get("error_rate_mean", 0.05))
            s = float(self._z.get("error_rate_std", 0.1))
            return z_score_to_unit(rate, m, s, self._clip_lo, self._clip_hi)
        return min_max_unit(rate, 0.0, 1.0, self._clip_lo, self._clip_hi)

    def _norm_global_rps(self, rps: float | None) -> float:
        if rps is None:
            return self._missing_scalar(pessimistic_high=True)
        if self._mode == "z_score":
            m = float(self._z.get("global_rps_mean", 1000.0))
            s = float(self._z.get("global_rps_std", 500.0))
            return z_score_to_unit(rps, m, s, self._clip_lo, self._clip_hi)
        return min_max_unit(rps, 0.0, self._ref_rps, self._clip_lo, self._clip_hi)

    def _default_now(self) -> float:
        times: list[float] = []
        for buf in (
            self._latency_buf,
            self._cpu_buf,
            self._mem_buf,
            self._queue_buf,
            self._rps_buf,
            self._request_buf,
        ):
            ts = buf.latest_timestamp()
            if ts is not None:
                times.append(ts)
        return max(times) if times else 0.0

    def build_vector(self, now: float | None = None) -> np.ndarray:
        """
        Assemble all 12 normalized features at simulation time `now` (seconds).

        If now is None, use the newest sample time so tests can skip a clock.
        """
        t_now = self._default_now() if now is None else float(now)
        lat_vals = self._latency_buf.values(t_now)
        p50 = percentile_linear(lat_vals, 50.0)
        p99 = percentile_linear(lat_vals, 99.0)
        n_p50, n_p99 = self._norm_latency_pair(p50, p99)

        cpu_raw = self._cpu_buf.mean(t_now)
        mem_raw = self._mem_buf.mean(t_now)
        q_raw = self._queue_buf.mean(t_now)
        n_cpu = self._norm_scalar_min_max(cpu_raw, 0.0, 1.0, pessimistic_high=True)
        n_mem = self._norm_scalar_min_max(mem_raw, 0.0, 1.0, pessimistic_high=True)
        n_queue = (
            self._missing_scalar(pessimistic_high=True)
            if q_raw is None
            else min_max_unit(
                q_raw,
                0.0,
                self._queue_cap,
                self._clip_lo,
                self._clip_hi,
            )
        )

        err = self._request_buf.error_rate(t_now)
        n_err = self._norm_error_rate(err)

        r_min = int(self._policy["retry"]["min"])
        r_max = int(self._policy["retry"]["max"])
        b_min = float(self._policy["backoff"]["min_multiplier"])
        b_max = float(self._policy["backoff"]["max_multiplier"])
        t_min = float(self._policy["timeout_ms"]["min"])
        t_max = float(self._policy["timeout_ms"]["max"])

        n_retry = min_max_unit(
            float(self.policy_context.retry_count),
            float(r_min),
            float(r_max),
            self._clip_lo,
            self._clip_hi,
        )
        n_back = min_max_unit(
            self.policy_context.backoff_multiplier,
            b_min,
            b_max,
            self._clip_lo,
            self._clip_hi,
        )
        n_tmo = min_max_unit(
            self.policy_context.timeout_ms,
            t_min,
            t_max,
            self._clip_lo,
            self._clip_hi,
        )

        rps_raw = self._rps_buf.mean(t_now)
        n_rps = self._norm_global_rps(rps_raw)

        vec = np.array(
            [
                n_cpu,
                n_mem,
                n_queue,
                n_p50,
                n_p99,
                n_err,
                n_retry,
                n_back,
                n_tmo,
                n_rps,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        assert vec.shape == (STATE_DIM,), vec.shape
        nan_fill = self._missing_scalar(pessimistic_high=True)
        vec = np.nan_to_num(
            vec,
            nan=nan_fill,
            posinf=self._clip_hi,
            neginf=self._clip_lo,
        )
        vec = np.clip(vec, self._clip_lo, self._clip_hi)
        return vec


def aggregator_from_config(root: dict[str, Any]) -> StateAggregator:
    """Create an aggregator from parsed YAML (needs observation + policy sections)."""
    return StateAggregator(observation_cfg=root["observation"], policy_cfg=root["policy"])


def build_state_vector(agg: StateAggregator, now: float | None = None) -> np.ndarray:
    """Readable alias for agg.build_vector(now)."""
    return agg.build_vector(now)
