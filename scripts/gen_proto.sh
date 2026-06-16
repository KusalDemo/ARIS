#!/usr/bin/env bash
# When we edit the .proto file, run this script so Python gets fresh auto-generated helper classes.
# Regenerate Python gRPC stubs from protos/. Run from repo root after editing .proto files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/aris-rl/src"
mkdir -p "$OUT"
python3 -m grpc_tools.protoc \
  -I "$ROOT/protos" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  --pyi_out="$OUT" \
  aris/policy/v1/policy.proto
touch "$OUT/aris/__init__.py" "$OUT/aris/policy/__init__.py" "$OUT/aris/policy/v1/__init__.py"
