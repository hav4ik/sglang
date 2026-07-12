#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${SGLANG_REPO_URL:-https://github.com/hav4ik/sglang.git}"
REF="${SGLANG_REF:-codex/flashinfer-attention-sink}"
CHECKOUT="${SGLANG_CHECKOUT:-/workspace/sglang}"
PYTHON="${PYTHON:-python3}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/sglang-cargo-target}"

sglang-sink-check-cuda

if [ -e "$CHECKOUT" ] && [ ! -d "$CHECKOUT/.git" ]; then
  echo "ERROR: $CHECKOUT exists but is not a git checkout" >&2
  exit 1
fi

if [ ! -d "$CHECKOUT/.git" ]; then
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$CHECKOUT"
elif [ -n "$(git -C "$CHECKOUT" status --porcelain)" ]; then
  echo "ERROR: refusing to replace a dirty checkout at $CHECKOUT" >&2
  exit 1
fi

git -C "$CHECKOUT" fetch --depth=1 origin "$REF"
git -C "$CHECKOUT" checkout --detach --force FETCH_HEAD

echo "Building the SGLang Rust gRPC extension for editable installation."
echo "No CUDA extension is built by this pip command."
export CARGO_TARGET_DIR
trap 'rm -rf "$CARGO_TARGET_DIR"' EXIT
"$PYTHON" -m pip install --no-cache-dir --no-deps --no-build-isolation \
  -e "$CHECKOUT/python"

# Avoid the checkout directory itself being resolved as a namespace package
# when bootstrap was invoked from its parent (for example, /workspace).
cd "$CHECKOUT"
"$PYTHON" - <<'PY'
import importlib.metadata as md
import inspect
import os
import re
import subprocess

import flashinfer
import sglang
import torch

checkout = os.path.realpath(os.environ.get("SGLANG_CHECKOUT", "/workspace/sglang"))
source = os.path.realpath(inspect.getfile(sglang))
assert source.startswith(checkout + os.sep), (source, checkout)
assert str(torch.version.cuda or "").startswith("12.8"), torch.version.cuda
assert md.version("flashinfer-python") == "0.6.14"
assert hasattr(flashinfer, "BatchAttentionWithAttentionSinkWrapper")

freeze = subprocess.check_output(
    [os.sys.executable, "-m", "pip", "list", "--format=freeze"], text=True
)
cu13 = [
    line
    for line in freeze.splitlines()
    if re.search(r"(cu13|cuda[-_]?13)", line.split("==", 1)[0], re.I)
]
assert not cu13, f"CUDA-13 packages found in cu128 image: {cu13}"
print("editable checkout ready", source)
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("flashinfer", md.version("flashinfer-python"))
PY

"$PYTHON" "$CHECKOUT/scripts/attention_sink/check_cuda_128.py"

rm -rf "$CARGO_TARGET_DIR"
trap - EXIT
