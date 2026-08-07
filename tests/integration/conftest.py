"""
Shared setup for integration tests: the ones that talk to real Docker containers.

Normal unit tests (aris-rl/tests) run in milliseconds and need nothing installed. The tests in this
folder are different: they boot the whole Compose stack, which takes minutes and needs Docker.

So they are OFF by default. Turn them on either way:

    ARIS_INTEGRATION=1 pytest tests/integration
    pytest -m integration

If you already started the stack yourself (`make compose-up-full`), set ARIS_COMPOSE_ALREADY_UP=1
so the fixture reuses it instead of building everything again.

On a cold machine the images have to build first, so the health wait below is generous. Override it
with ARIS_COMPOSE_HEALTH_TIMEOUT (seconds) on slow hardware.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# This file lives at <repo>/tests/integration/conftest.py, so two levels up is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "infra" / "docker" / "docker-compose.yml"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip anything marked `integration` unless the operator explicitly opted in."""
    markexpr = getattr(config.option, "markexpr", "") or ""
    if markexpr and "integration" in markexpr:
        # The person ran `pytest -m integration`, so they clearly want these tests.
        return

    skip_no_env = pytest.mark.skip(
        reason="Set ARIS_INTEGRATION=1 to run integration tests, or run: pytest -m integration",
    )
    if _env_truthy("ARIS_INTEGRATION"):
        return
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_no_env)


@pytest.fixture(scope="module")
def compose_stack() -> Iterator[None]:
    """
    Bring the Compose testbed up once for the whole test module, then tear it down.

    "Healthy" here means the two ends of the chain answer: the echo app (simplest service) and the
    aggregator (which only works once Toxiproxy and its downstreams are wired). If those two are
    fine, the rest of the stack is almost certainly fine too.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")

    # CI starts Compose once in the workflow, then runs these tests against the live stack.
    if _env_truthy("ARIS_COMPOSE_ALREADY_UP"):
        yield
        return

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        timeout=600,
    )

    # Toxiproxy is wired and echo/cpu are healthy, so its compose healthcheck alone allows 40s.
    timeout_s = float(os.environ.get("ARIS_COMPOSE_HEALTH_TIMEOUT", "180"))
    deadline = time.time() + timeout_s
    ok = False
    while time.time() < deadline:
        try:
            # -sf = silent, and fail (non-zero exit) on HTTP errors instead of printing the body.
            # The per-curl timeouts matter: without them a hung container would eat the whole budget.
            echo_ok = subprocess.run(
                ["curl", "-sf", "--max-time", "5", "http://127.0.0.1:18001/health/live"],
                capture_output=True,
                timeout=10,
            )
            aggregator_ok = subprocess.run(
                ["curl", "-sf", "--max-time", "10", "http://127.0.0.1:18003/aggregate"],
                capture_output=True,
                timeout=15,
            )
            if echo_ok.returncode == 0 and aggregator_ok.returncode == 0:
                ok = True
                break
        except (OSError, subprocess.TimeoutExpired):
            # Containers still starting: keep polling until the deadline.
            pass
        # Poll every 2s: often enough to feel instant, rare enough to keep the logs readable.
        time.sleep(2.0)

    if not ok:
        # Print recent container logs so a CI failure is debuggable without re-running locally.
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "logs", "--tail", "80"],
            cwd=REPO_ROOT,
        )
        pytest.fail(f"Compose stack did not become healthy within {timeout_s:.0f}s")

    yield

    # `down -v` also removes volumes so the next run starts from a clean slate.
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
        cwd=REPO_ROOT,
        timeout=120,
    )
