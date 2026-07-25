"""Training helpers (offline/online entrypoints)."""

from aris_rl.training.offline import run_offline_training
from aris_rl.training.online import run_online_training

__all__ = ["run_offline_training", "run_online_training"]
