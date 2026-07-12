#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${RESULTS:-$ROOT/attention-sink-results/$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"
PROFILE="${PROFILE:-kernel}"
mkdir -p "$RESULTS"
rm -f "$RESULTS/completion.json"
cd "$ROOT"

write_completion() {
  "$PYTHON" - "$RESULTS/completion.json" "$PROFILE" <<'PY'
import json, sys, time
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "passed",
    "profile": sys.argv[2],
    "completed_unix_s": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY
}

"$PYTHON" scripts/attention_sink/check_cuda_128.py \
  2> >(tee "$RESULTS/cuda-12.8-gate.stderr" >&2) \
  | tee "$RESULTS/cuda-12.8-gate.json"

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

if [ "${SKIP_KERNEL_TESTS:-0}" != 1 ]; then
  "$PYTHON" -m pytest -q \
    test/registered/unit/model_loader/test_flash_rl_attention_sinks.py \
    test/registered/unit/model_loader/test_attention_sink_checkpoint_tools.py \
    test/registered/unit/layers/test_flashinfer_attention_sinks.py \
    test/registered/unit/utils/test_weight_checker.py::TestHandle::test_sink_checksum_action_filters_other_parameters \
    test/registered/attention/test_flashinfer_attention_sink.py \
    | tee "$RESULTS/registered-tests.txt"

  SINK_LONG_MAX_CONTEXT="${SINK_LONG_MAX_CONTEXT:-131072}" \
    "$PYTHON" -m pytest -q -s \
    test/manual/attention/test_attention_sink_hardware.py \
    | tee "$RESULTS/long-context-tests.txt"

fi

if [ "$PROFILE" = kernel ]; then
  write_completion
  echo "kernel validation complete: $RESULTS"
  exit 0
fi

MODEL="${MODEL:?set MODEL for server or rl profile}"
TP="${TP:-1}"
PORT="${PORT:-30000}"
BACKENDS="${BACKENDS:-triton flashinfer}"
KV_CACHE_DTYPES="${KV_CACHE_DTYPES:-auto fp8_e4m3}"
QUANTIZATIONS="${QUANTIZATIONS:-fp8}"

cleanup_server() {
  kill "${monitor_pid:-}" "${rss_monitor_pid:-}" 2>/dev/null || true
  if [ -n "${server_pid:-}" ]; then
    kill -- -"$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! ps -eo pgid=,stat= | awk -v target="$server_pid" \
        '$1 == target && $2 !~ /^Z/ { found = 1 } END { exit !found }'; then
        break
      fi
      sleep 1
    done
    if ps -eo pgid=,stat= | awk -v target="$server_pid" \
      '$1 == target && $2 !~ /^Z/ { found = 1 } END { exit !found }'; then
      kill -KILL -- -"$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  if [ -n "${monitor_pid:-}" ]; then
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [ -n "${rss_monitor_pid:-}" ]; then
    wait "$rss_monitor_pid" 2>/dev/null || true
  fi
}

"$PYTHON" scripts/attention_sink/validate_checkpoint.py "$MODEL" \
  --output "$RESULTS/checkpoint-a.json"
if [ "$PROFILE" = rl ]; then
  "$PYTHON" scripts/attention_sink/validate_checkpoint.py \
    "${RELOAD_MODEL:?set RELOAD_MODEL for rl profile}" \
    --output "$RESULTS/checkpoint-b.json"
fi

for quantization in $QUANTIZATIONS; do
for kv_cache_dtype in $KV_CACHE_DTYPES; do
for backend in $BACKENDS; do
  label="$backend-$kv_cache_dtype-$quantization"
  log="$RESULTS/server-$label.log"
  quantization_args=()
  cuda_graph_args=()
  load_format=auto
  if [ "$quantization" != none ]; then
    quantization_args=(--quantization "$quantization")
    load_format=flash_rl
  fi
  if [ -n "${CUDA_GRAPH_MAX_BS_DECODE:-}" ]; then
    cuda_graph_args+=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE")
  fi
  if [ -n "${CUDA_GRAPH_MAX_BS_PREFILL:-}" ]; then
    cuda_graph_args+=(--cuda-graph-max-bs-prefill "$CUDA_GRAPH_MAX_BS_PREFILL")
  fi
  if ! "$PYTHON" - "$PORT" <<'PY'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
  then
    echo "ERROR: port $PORT is already in use" >&2
    exit 1
  fi
  setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size "$TP" \
    --host 127.0.0.1 --port "$PORT" \
    --attention-backend "$backend" --page-size 1 \
    "${quantization_args[@]}" --load-format "$load_format" \
    "${cuda_graph_args[@]}" \
    --kv-cache-dtype "$kv_cache_dtype" \
    --context-length "${CONTEXT_LEN:-131328}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
    --mem-fraction-static "${MEMFRAC:-0.80}" \
    --disable-radix-cache --skip-tokenizer-init \
    >"$log" 2>&1 &
  server_pid=$!
  gpu_query_args=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_query_args=(-i "$CUDA_VISIBLE_DEVICES")
  fi
  nvidia-smi "${gpu_query_args[@]}" \
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
  trap cleanup_server EXIT
  ready=0
  for _ in $(seq 1 360); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -200 "$log"; exit 1
    fi
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 5
  done
  [ "$ready" = 1 ] || { tail -200 "$log"; exit 1; }

  probe_args=(
    --url "http://127.0.0.1:$PORT"
    --model "$MODEL"
    --output "$RESULTS/probe-$label.json"
    --lengths "${PROBE_LENGTHS:-128,4095,4096,4097,16384}"
    --output-tokens "${PROBE_OUTPUT_TOKENS:-4}"
  )
  if [ "$PROFILE" = rl ]; then
    probe_args+=(--reload-model "$RELOAD_MODEL" --reload-load-format "$load_format")
    if [ "$load_format" = flash_rl ] && [ "${WARM_RELOAD:-1}" = 1 ]; then
      probe_args+=(--warm-reload)
    fi
    if [ "${REQUIRE_RELOAD_CHANGE:-1}" = 1 ]; then
      probe_args+=(--require-reload-change)
    fi
  fi
  if ! "$PYTHON" scripts/attention_sink/probe_server.py "${probe_args[@]}"; then
    echo "probe failed; last 250 lines from $log:" >&2
    tail -250 "$log" >&2 || true
    exit 1
  fi
  cleanup_server
  trap - EXIT
done
if [ -f "$RESULTS/probe-triton-$kv_cache_dtype-$quantization.json" ] && \
   [ -f "$RESULTS/probe-flashinfer-$kv_cache_dtype-$quantization.json" ]; then
  "$PYTHON" scripts/attention_sink/compare_probes.py \
    "$RESULTS/probe-triton-$kv_cache_dtype-$quantization.json" \
    "$RESULTS/probe-flashinfer-$kv_cache_dtype-$quantization.json" \
    | tee "$RESULTS/backend-comparison-$kv_cache_dtype-$quantization.json"
fi
done
done

if [ "$PROFILE" = rl ] && [ "${RUN_LIVE_SINK_UPDATE:-1}" = 1 ]; then
  for quantization in ${LIVE_QUANTIZATIONS:-fp8}; do
  for kv_cache_dtype in ${LIVE_KV_CACHE_DTYPES:-auto}; do
  for backend in $BACKENDS; do
    "$PYTHON" scripts/attention_sink/validate_live_sink_update.py \
      --model "$MODEL" --tp "$TP" --attention-backend "$backend" \
      --quantization "$quantization" --kv-cache-dtype "$kv_cache_dtype" \
      --context-length "${CONTEXT_LEN:-131328}" \
      --lengths "${LIVE_PROBE_LENGTHS:-128,4097,16384}" \
      --output-tokens "${PROBE_OUTPUT_TOKENS:-4}" \
      --mem-fraction-static "${MEMFRAC:-0.80}" \
      --output "$RESULTS/live-$backend-$kv_cache_dtype-$quantization.json"
  done
  done
  done
fi

write_completion
echo "hardware validation complete: $RESULTS"
