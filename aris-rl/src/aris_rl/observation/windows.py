"""
Rolling time windows: only keep measurements from the last N seconds.

Think of it like a attendance sheet that only lists people who showed up in the
last class period — old rows fall off when time moves forward. That way "error rate
in the last 30 seconds" really means the last 30 seconds, not since the dawn of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimedFloatBuffer:
    """Stores (timestamp, number) pairs for things like latency or CPU usage."""

    window_s: float
    _items: list[tuple[float, float]] = field(default_factory=list)

    def append(self, t: float, value: float) -> None:
        # Not-a-number would break averages and percentiles, so we ignore it.
        if not np.isfinite(value):
            return
        self._items.append((t, float(value)))
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        if self._items and self._items[0][0] < cutoff:
            self._items = [x for x in self._items if x[0] >= cutoff]

    def values(self, now: float) -> np.ndarray:
        """All values still inside the window, in the order they arrived."""
        self._prune(now)
        if not self._items:
            return np.array([], dtype=np.float64)
        return np.array([v for _, v in self._items], dtype=np.float64)

    def mean(self, now: float) -> float | None:
        """Average of values in the window, or None if the window is empty."""
        arr = self.values(now)
        if arr.size == 0:
            return None
        return float(np.mean(arr))

    def latest_timestamp(self) -> float | None:
        """Newest sample time, if we have any samples."""
        if not self._items:
            return None
        return float(self._items[-1][0])


@dataclass
class TimedRequestBuffer:
    """
    One row per request: did it fail?

    Error rate = (failed requests) / (all requests) still inside the window.
    """

    window_s: float
    _items: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, t: float, is_error: bool) -> None:
        self._items.append((t, is_error))
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        if self._items and self._items[0][0] < cutoff:
            self._items = [x for x in self._items if x[0] >= cutoff]

    def error_rate(self, now: float) -> float | None:
        """Fraction of errors in the window, or None if no requests were recorded."""
        self._prune(now)
        if not self._items:
            return None
        errors = sum(1 for _, err in self._items if err)
        return errors / len(self._items)

    def latest_timestamp(self) -> float | None:
        if not self._items:
            return None
        return float(self._items[-1][0])
