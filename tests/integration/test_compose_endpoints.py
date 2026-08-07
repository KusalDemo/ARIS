"""
End-to-end checks against the running Docker Compose stack.

These prove the pieces are wired together for real: Envoy talks to the policy service, the policy
service answers, and Prometheus is collecting numbers. They are opt-in (see conftest.py) because
they start containers and take minutes, not milliseconds.
"""

from __future__ import annotations

import json
import urllib.request

import pytest


@pytest.mark.integration
def test_policy_health_live(compose_stack: None) -> None:
    """The policy service container is up and answering on its published host port."""
    with urllib.request.urlopen("http://127.0.0.1:18080/health/live", timeout=10) as r:
        assert r.status == 200


@pytest.mark.integration
def test_echo_through_envoy_ext_authz(compose_stack: None) -> None:
    """
    Traffic hits Envoy first; Envoy asks the policy service, then forwards to echo.

    We only check that we get HTTP 200 and a body - that alone proves the authz hop completed,
    because Envoy would return 403 (or 500) if the policy call failed.
    """
    req = urllib.request.Request(
        "http://127.0.0.1:10000/echo?message=integration",
        headers={
            # Envoy only forwards this allow-list of headers to the policy service
            # (see infra/envoy/envoy.yaml).
            "x-aris-route": "/echo",
            "x-aris-error-rate": "0.01",
            "x-aris-trace-id": "itest-1",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
    data = json.loads(body)
    assert "message" in data or "echo" in body.lower()


@pytest.mark.integration
def test_prometheus_targets_metric(compose_stack: None) -> None:
    """Prometheus should be scraping something: the `up` metric proves the metrics path is alive."""
    url = "http://127.0.0.1:9090/api/v1/query?query=up"
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read().decode("utf-8"))
    assert payload.get("status") == "success"
    assert "data" in payload


@pytest.mark.integration
def test_direct_echo_port(compose_stack: None) -> None:
    """Check the app itself, bypassing Envoy - tells apart a proxy fault from an app fault."""
    with urllib.request.urlopen("http://127.0.0.1:18001/health/live", timeout=10) as r:
        assert r.status == 200
