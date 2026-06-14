"""Console entry points"""

from __future__ import annotations

import sys


def _stub(name: str) -> None:
    print(f"{name}: not implemented yet..", file=sys.stderr)
    raise SystemExit(2)


def main_train() -> None:
    _stub("aris-train")


def main_export() -> None:
    _stub("aris-export")


def main_validate_config() -> None:
    _stub("aris-validate-config")


def main_eval() -> None:
    _stub("aris-eval")


def main_benchmark() -> None:
    _stub("aris-benchmark")
