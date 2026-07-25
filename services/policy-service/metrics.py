"""Prometheus counters/gauges for the policy hot path (scraped from `/metrics`)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

DECISIONS_TOTAL = Counter(
    "aris_policy_decisions_total",
    "How many safeguard decisions were computed.",
    labelnames=("route",),
)

OVERRIDES_TOTAL = Counter(
    "aris_safeguard_overrides_total",
    "How many times a safeguard changed or blocked something (reason as label).",
    labelnames=("route", "reason"),
)

FREEZE_ACTIVE = Gauge(
    "aris_policy_freeze_active",
    "Whether the circuit breaker is holding the last safe policy for a route (1 = frozen).",
    labelnames=("route",),
)

DECISION_LATENCY_SECONDS = Histogram(
    "aris_policy_decision_latency_seconds",
    "Wall time spent producing one policy answer (seconds).",
    buckets=(0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
