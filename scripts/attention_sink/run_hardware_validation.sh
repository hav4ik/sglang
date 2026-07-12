#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${RESULTS:-$ROOT/attention-sink-results/$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"
PROFILE="${PROFILE:-kernel}"
mkdir -p "$RESULTS"
cd "$ROOT"

nvidia-smi -q >"$RESULTS/nvidia-smi.txt"
git rev-parse HEAD >"$RESULTS/git-revision.txt"
git status --short >"$RESULTS/git-status.txt"
"$PYTHON" - <<'PY' >"$RESULTS/environment.json"
import json, platform
import torch
try:
    import flashinfer
    flashinfer_version = flashinfer.__version__
except Exception as exc:
    flashinfer_version = f"unavailable: {exc}"
print(json.dumps({
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
    "flashinfer": flashinfer_version,
}, indent=2))
PY

"$PYTHON" -m pytest -q \
  test/registered/unit/model_loader/test_flash_rl_attention_sinks.py \
  test/registered/unit/layers/test_flashinfer_attention_sinks.py \
  test/registered/attention/test_flashinfer_attention_sink.py \
  | tee "$RESULTS/registered-tests.txt"

SINK_LONG_MAX_CONTEXT="${SINK_LONG_MAX_CONTEXT:-131072}" \
  "$PYTHON" -m pytest -q -s \
  test/manual/attention/test_attention_sink_hardware.py \
  | tee "$RESULTS/long-context-tests.txt"

if [ "$PROFILE" = kernel ]; then
  echo "kernel validation complete: $RESULTS"
  exit 0
fi

MODEL="${MODEL:?set MODEL for server or rl profile}"
TP="${TP:-1}"
PORT="${PORT:-30000}"
BACKENDS="${BACKENDS:-triton flashinfer}"
KV_CACHE_DTYPES="${KV_CACHE_DTYPES:-auto fp8_e4m3}"
"$PYTHON" scripts/attention_sink/validate_checkpoint.py "$MODEL" \
  --output "$RESULTS/checkpoint-a.json"
if [ "$PROFILE" = rl ]; then
  "$PYTHON" scripts/attention_sink/validate_checkpoint.py \
    "${RELOAD_MODEL:?set RELOAD_MODEL for rl profile}" \
    --output "$RESULTS/checkpoint-b.json"
fi

for kv_cache_dtype in $KV_CACHE_DTYPES; do
for backend in $BACKENDS; do
  label="$backend-$kv_cache_dtype"
  log="$RESULTS/server-$label.log"
  setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size "$TP" \
    --host 127.0.0.1 --port "$PORT" \
    --attention-backend "$backend" --page-size 1 \
    --quantization fp8 --load-format flash_rl \
    --kv-cache-dtype "$kv_cache_dtype" \
    --context-length "${CONTEXT_LEN:-131072}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
    --mem-fraction-static "${MEMFRAC:-0.80}" \
    --disable-radix-cache --skip-tokenizer-init \
    >"$log" 2>&1 &
  server_pid=$!
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv -l 1 >"$RESULTS/gpu-$label.csv" &
  monitor_pid=$!
  (
    while kill -0 "$server_pid" 2>/dev/null; do
      date --iso-8601=ns
      ps -eo pid,pgid,rss,vsz,comm --no-headers \
        | awk -v pgid="$server_pid" '$2 == pgid'
      sleep 1
    done
  ) >"$RESULTS/process-memory-$label.txt" &
  rss_monitor_pid=$!
  trap 'kill "$monitor_pid" "$rss_monitor_pid" 2>/dev/null || true; kill -- -"$server_pid" 2>/dev/null || true' EXIT
  ready=0
  for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
      ready=1; break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -200 "$log"; exit 1
    fi
    sleep 5
  done
  [ "$ready" = 1 ] || { tail -200 "$log"; exit 1; }

  probe_args=(
    --url "http://127.0.0.1:$PORT"
    --model "$MODEL"
    --output "$RESULTS/probe-$label.json"
    --lengths "${PROBE_LENGTHS:-128,4095,4096,4097,16384}"
  )
  if [ "$PROFILE" = rl ]; then
    probe_args+=(--reload-model "$RELOAD_MODEL")
  fi
  "$PYTHON" scripts/attention_sink/probe_server.py "${probe_args[@]}"
  kill "$monitor_pid" "$rss_monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" "$rss_monitor_pid" 2>/dev/null || true
  kill -- -"$server_pid" 2>/dev/null || true
  wait "$server_pid" || true
  trap - EXIT
done
done

for kv_cache_dtype in $KV_CACHE_DTYPES; do
if [ -f "$RESULTS/probe-triton-$kv_cache_dtype.json" ] && \
   [ -f "$RESULTS/probe-flashinfer-$kv_cache_dtype.json" ]; then
  "$PYTHON" scripts/attention_sink/compare_probes.py \
    "$RESULTS/probe-triton-$kv_cache_dtype.json" \
    "$RESULTS/probe-flashinfer-$kv_cache_dtype.json" \
    | tee "$RESULTS/backend-comparison-$kv_cache_dtype.json"
fi
done

echo "hardware validation complete: $RESULTS"
