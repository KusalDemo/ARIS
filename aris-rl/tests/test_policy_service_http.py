"""Smoke the FastAPI app the same way Docker would (no network needed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def client() -> TestClient:
    os.environ["ARIS_CONFIG"] = str(REPO_ROOT / "configs" / "benchmark-static.yaml")
    from main import app  # noqa: E402

    with TestClient(app) as tc:
        yield tc


def test_health_and_metrics(client: TestClient) -> None:
    assert client.get("/health/live").text == "ok"
    m = client.get("/metrics")
    assert m.status_code == 200
    assert b"aris_policy_decisions_total" in m.content
    assert b"aris_policy_decision_latency_seconds" in m.content
    assert b"aris_safeguard_overrides_total" in m.content


def test_decide_logs_shape(client: TestClient) -> None:
    resp = client.post(
        "/decide",
        json={
            "route": "/demo",
            "error_rate": 0.0,
            "suggested": {"retry": 2, "backoff_multiplier": 1.0, "timeout_ms": 2000.0},
            "trace_id": "abc123",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retry"] == 2
    assert body["frozen_active"] is False


def test_ext_authz_via_client_path_like_envoy(client: TestClient) -> None:
    """Envoy http ext_authz uses the downstream path on the policy cluster, not /envoy/ext_authz."""
    resp = client.get(
        "/echo?message=integration",
        headers={
            "x-aris-route": "/echo",
            "x-aris-error-rate": "0.01",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-aris-retry") is not None
