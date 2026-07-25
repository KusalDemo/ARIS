"""
Scalar reward from one step: reward good success, penalize slow paths and overload.

Weights live in YAML under reward.weights; cascade_active adds a fixed extra penalty.
"""

from __future__ import annotations

from typing import Any


def compute_reward(
    *,
    success_rate: float,
    avg_latency_ms: float,
    retry_overhead: float,
    overload_penalty: float,
    cascade_active: bool,
    reward_cfg: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    w = reward_cfg["weights"]
    w_s = float(w["success_rate"])
    w_l = float(w["latency"])
    w_r = float(w["retry_overhead"])
    w_o = float(w["overload_penalty"])
    c_scale = float(reward_cfg["cascade_penalty_scale"])

    term_s = w_s * float(success_rate)
    term_l = w_l * (float(avg_latency_ms) / 1000.0)
    term_r = w_r * float(retry_overhead)
    term_o = w_o * float(overload_penalty)
    cascade_hit = c_scale if cascade_active else 0.0
    total = term_s - term_l - term_r - term_o - cascade_hit

    breakdown = {
        "term_success": term_s,
        "term_latency": term_l,
        "term_retry": term_r,
        "term_overload": term_o,
        "term_cascade": cascade_hit,
    }
    return float(total), breakdown
