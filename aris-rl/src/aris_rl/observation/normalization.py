"""
Squash raw measurements into roughly 0–1 so the neural network gets steady inputs.

min_max: map a known low/high range into [clip_lo, clip_hi] (like grading on a curve
with fixed min and max scores).

z_score: ask "how unusual is this vs a typical value?" then squash that into the same range.

percentile: pick a value from a sorted list (p50 = median, p99 = almost the worst).
"""

from __future__ import annotations

import numpy as np


def clip_to_range(x: float, lo: float, hi: float) -> float:
    """Keep x between lo and hi — nothing below lo, nothing above hi."""
    return float(np.clip(x, lo, hi))


def min_max_unit(v: float, v_min: float, v_max: float, clip_lo: float, clip_hi: float) -> float:
    """Stretch v from [v_min, v_max] into [clip_lo, clip_hi], then clip again for safety."""
    if v_max <= v_min:
        return clip_to_range((clip_lo + clip_hi) / 2.0, clip_lo, clip_hi)
    u = (v - v_min) / (v_max - v_min)
    out = clip_lo + u * (clip_hi - clip_lo)
    return clip_to_range(out, clip_lo, clip_hi)


def z_score_to_unit(
    v: float,
    mean: float,
    std: float,
    clip_lo: float,
    clip_hi: float,
) -> float:
    """
    Turn "how many standard deviations from mean" into a 0–1 style number.

    We cap at ±3σ so one crazy spike does not blow up the whole vector.
    """
    s = max(float(std), 1e-9)
    t = (float(v) - float(mean)) / s
    t = float(np.clip(t, -3.0, 3.0))
    u = (t + 3.0) / 6.0
    out = clip_lo + u * (clip_hi - clip_lo)
    return clip_to_range(out, clip_lo, clip_hi)


def percentile_linear(values: np.ndarray, q: float) -> float | None:
    """q-th percentile (q from 0 to 100), or None when there is no data."""
    if values.size == 0:
        return None
    return float(np.percentile(values, q, method="linear"))
