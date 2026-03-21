#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required and was not found in PATH."
  exit 1
fi

TARGET_SCRIPT="${1:-benchmarks/bench_softmax.py}"
shift || true

nsys profile --trace=cuda,nvtx,osrt python "${TARGET_SCRIPT}" --device cuda "$@"

