"""Tests for optional Prometheus export from replay storage and training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from aris_rl.config import load_config, validate_config_keys, validate_prometheus_config
from aris_rl.envs import ArisMicroserviceEnv
from aris_rl.envs.wrappers import FlattenMultiDiscreteActions
from aris_rl.monitoring.training_metrics import (
    TrainingMetricsExporter,
    maybe_start_training_metrics,
)
from aris_rl.replay_storage import open_storage
from aris_rl.training.metrics_callback import TrainingMetricsCallback

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "aris.default.yaml"


def test_validate_prometheus_config_accepts_default() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_prometheus_config(data["prometheus"])


def test_validate_prometheus_config_rejects_bad_port() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["scrape_port"] = 70000
    with pytest.raises(ValueError, match="65535"):
        validate_prometheus_config(data["prometheus"])


def test_replay_metrics_track_flushes_and_rotation(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["transitions_per_shard"] = 3
    data["replay_buffer"]["max_shards"] = 2
    validate_config_keys(data)

    reg = CollectorRegistry()
    exporter = TrainingMetricsExporter(data, registry=reg)

    env = ArisMicroserviceEnv(data, algorithm="dqn")
    wenv = FlattenMultiDiscreteActions(env)
    store = open_storage(tmp_path / "rb", data, env=env, metrics_exporter=exporter)
    obs, _ = wenv.reset(seed=0)
    for _ in range(9):
        a = int(wenv.action_space.sample())
        obs2, _, _, _, _ = wenv.step(a)
        store.append(obs, obs2, np.array([a], dtype=np.int32), 0.0, False)
        obs = obs2
    store.flush()

    text = generate_latest(reg).decode("utf-8")
    assert "aris_replay_shard_flushes_total 3.0" in text
    assert "aris_replay_shards_removed_total 1.0" in text


def test_training_metrics_callback_records_reward() -> None:
    exporter = MagicMock()
    cb = TrainingMetricsCallback(exporter, algorithm="DQN")
    cb.locals = {"rewards": [0.5, 1.5]}
    assert cb._on_step() is True
    exporter.record_env_step.assert_called_once()

    cb2 = TrainingMetricsCallback(exporter, algorithm="DQN")
    cb2.locals = {}
    assert cb2._on_step() is True


def test_maybe_start_training_metrics_off() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    assert data["prometheus"]["enabled"] is False
    assert maybe_start_training_metrics(data) is None


def test_maybe_start_training_metrics_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import aris_rl.monitoring.training_metrics as tm

    calls: list[tuple[int, str]] = []

    def fake_start(port: int, addr: str = "0.0.0.0", registry: object | None = None) -> None:
        calls.append((port, addr))

    monkeypatch.setattr(tm, "start_http_server", fake_start)
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["enabled"] = True
    validate_config_keys(data)
    out = maybe_start_training_metrics(data)
    assert out is not None
    assert calls


def test_training_metrics_exporter_sync_and_steps(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    validate_config_keys(data)
    reg = CollectorRegistry()
    ex = TrainingMetricsExporter(data, registry=reg)
    storage = MagicMock()
    storage.total_transitions = 10
    storage.pending_count = 2
    ex.sync_replay_from_storage(storage)
    ex.on_replay_flush()
    ex.on_replay_shard_removed()
    ex.record_env_step(algorithm="dqn", reward=1.0)
    ex.record_env_step(algorithm="dqn", reward=-1.0)


def test_training_metrics_start_http_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import aris_rl.monitoring.training_metrics as tm

    monkeypatch.setattr(tm, "start_http_server", lambda *a, **k: None)
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["enabled"] = True
    validate_config_keys(data)
    reg = CollectorRegistry()
    ex = TrainingMetricsExporter(data, registry=reg)
    ex.start_http_server(data)
