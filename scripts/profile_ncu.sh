#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu is required and was not found in PATH."
  exit 1
fi

TARGET_SCRIPT="${1:-benchmarks/bench_softmax.py}"
shift || true

ncu --set full --target-processes all python "${TARGET_SCRIPT}" --device cuda "$@"

