#!/usr/bin/env bash
set -euo pipefail

CHECKOUT="${SGLANG_CHECKOUT:-/workspace/sglang}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$CHECKOUT/.git" ]; then
  echo "ERROR: run sglang-sink-bootstrap before testing" >&2
  exit 1
fi

cd "$CHECKOUT"
export PROFILE=kernel
export PYTHON
export SINK_LONG_MAX_CONTEXT="${SINK_LONG_MAX_CONTEXT:-131072}"
export RESULTS="${RESULTS:-/workspace/results/$(date +%Y%m%d-%H%M%S)}"

exec scripts/attention_sink/run_hardware_validation.sh
