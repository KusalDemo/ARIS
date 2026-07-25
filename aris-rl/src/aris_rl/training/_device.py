"""Pick CPU or GPU for PyTorch without surprising the operator."""

from __future__ import annotations


def resolve_training_device(configured: str) -> str:
    """Return cpu, cuda, or auto-detect when configured is 'auto'."""
    if configured == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return configured
