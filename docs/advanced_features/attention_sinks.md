# OLMo3 Attention Sinks

This branch adds inference and RL reload support for OLMo3 models with one learned
attention-sink logit per query head. The target artifact is
`chankhavu/yccchen-olmo3-deploy` (`Olmo3SinkForCausalLM`). Yi-Chia Chen's Triton
path remains the rollout default; FlashInfer is an explicit, release-gated option.
The audited model revision is `39beac79e6857df6d8a0dc27210f5affa4031c92`.

## Semantics

For query head `h`, sink logit `s_h` adds a zero-valued softmax entry:

```text
normalizer = exp(s_h) + sum_i exp(q_h k_i / sqrt(d))
output_h   = sum_i exp(q_h k_i / sqrt(d)) * v_i / normalizer
```

The eager reference, SGLang Triton kernels, and FlashInfer's dedicated
`BatchAttentionWithAttentionSinkWrapper` implement this definition. Passing
`sinks=` to an ordinary FlashInfer paged wrapper is not sufficient and may be
silently ignored, so sink models exclusively use the dedicated wrapper.

FlashInfer 0.6.14's dedicated wrapper gives BF16-KV and FP8-KV sink kernels the
same JIT cache identity because that identity omits the KV dtype. SGLang uses
FlashInfer's own dtype-complete sink URI helper when constructing the otherwise
identical custom kernel, preventing a shared BF16 cache entry from being reused
as FP8 (or vice versa).

## Checkpoint Contract

The 32B deploy config has 64 layers, 40 query heads, 8 KV heads, hidden size
5120, head dimension 128, and `sliding_window=4096`. It must select
`Olmo3SinkForCausalLM`.

Each layer carries these BF16 tensors:

| Tensor suffix | Shape |
|---|---:|
| `self_attn.sinks` | `[40]` |
| `self_attn.q_norm.weight` | `[5120]` |
| `self_attn.k_norm.weight` | `[1024]` |
| `post_attention_layernorm.weight` | `[5120]` |
| `post_feedforward_layernorm.weight` | `[5120]` |

The complete target checkpoint contains 771 weights: embeddings, LM head, final
norm, and 12 tensors per layer. Initial loading and FlashRL reload reject any
missing weight before commit. Sink parameters are TP-sharded like query heads and
stored as FP32 at runtime because FlashInfer requires FP32 sink logits.

```bash
python scripts/attention_sink/validate_checkpoint.py /models/yccchen-olmo3-deploy
```

The OPD writer performs the same gate before publishing and writes
`attention_sinks.json` with the sink checksum and value range.

## Backend Matrix

| Path | Triton | FlashInfer |
|---|---|---|
| Initial prefill | Supported | Supported, paged path |
| Cached/chunked extend | Supported | Supported |
| Decode | Supported | Supported |
| Full attention and OLMo3 SWA | Supported | Supported |
| CUDA graph decode | Supported | Supported |
| FP8 weights with BF16 KV | Supported | Supported |
| FP8 E4M3 KV | Supported on capable GPUs | H100/B200 release gate |
| Page size `1` | Supported | Supported |
| Page size greater than `1` | Backend-dependent | Rejected for sink models |
| DCP | Rejected for sink models | Rejected for sink models |
| Multi-item scoring | Supported by ordinary Triton | Rejected for sink models |
| Custom/tree masks | Supported by ordinary Triton | Rejected for sink models |
| DFlash/EAGLE draft | Deferred | Deferred |

The target's `sliding_window=4096` becomes SGLang's inclusive
`window_left=4095`. Hardware tests cover sequence lengths 4095, 4096, and 4097.

## RL Update Protocol

FlashRL reload has a prepare phase and a commit phase:

1. Materialize the incoming checkpoint and require all 771 target names.
2. Stage FP8 matrices in GPU scratch and unquantized weights on CPU.
3. Quantize/load every staged weight and validate sinks, shape, and dtype.
4. Copy staged values into the original graph-stable parameter storage.

Failures before commit restore all original parameter pointers. Commit consists of
prevalidated copies into existing storage, but a device fault during commit cannot
be rolled back; the server reports uncertain state and must remain paused.

OPD uses this replica barrier:

```text
save and validate checkpoint
-> abort active generation on every replica
-> reload every replica with KV flush
-> verify every response
-> resume every replica
-> advance the global version
```

An aborted rollout retries the same prompt after the pause gate opens, so no
trajectory spans a weight boundary. If any stage fails, the global version does
not advance and paused replicas remain paused.

FP8 qualification uses `cold A0 -> reload A1 -> reload B -> reload A2` and
requires strict A1/A2 parity. Cold startup quantizes row-parallel weights after
TP input-column slicing, while FlashRL quantizes the global row before slicing;
their per-channel scales can differ even for the same BF16 checkpoint. The
cold-A0/A1 delta is retained in the result report, not hidden by a looser
tolerance. A1/A2 exercise the same production reload path and must match. Sink
checksums are compared exactly on every TP rank across A1, B, and A2.

## Local Tests

```bash
python -m pytest -q \
  test/registered/unit/model_loader/test_flash_rl_attention_sinks.py \
  test/registered/unit/model_loader/test_attention_sink_checkpoint_tools.py \
  test/registered/unit/layers/test_flashinfer_attention_sinks.py \
  test/registered/attention/test_flashinfer_attention_sink.py
```

These cover target 40:8 GQA, head dimension 128, prefill, cached extend, decode,
full attention, short-window SWA, BF16 KV, FP8 KV on supported GPUs, and direct
FlashInfer CUDA graph replay after sink mutation at batch sizes 1, 2, and 8.

## H100/B200 Qualification

The CUDA 12.8 development image contains all Python, CUDA, Rust, and protobuf
dependencies. On a fresh node, pull the image and activate this branch:

```bash
docker pull chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/cache:/cache" -v "$PWD/workspace:/workspace" \
  -v /shared/models:/models \
  chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128
sglang-sink-bootstrap
sglang-sink-check-cuda
sglang-sink-tests
```

`sglang-sink-bootstrap` compiles only the small Rust gRPC extension. The first
test run may JIT architecture-specific FlashInfer and Triton CUDA kernels into
the mounted `/cache`; subsequent runs reuse that cache.
`sglang-sink-check-cuda` fails on nvcc, libcudart, Torch, native wheel tags,
Debian packages, or filesystem toolkits newer than CUDA 12.8. It reports but
allows `cuda-python`/`cuda-bindings` 12.9.4 because they are Python API wrappers
required by Torch's cu128 wheel, not toolkit/runtime libraries. It also permits
only the checksum-pinned `sglang-kernel==0.4.4+cu129` native-wheel exception.
That wheel contains sm90/sm100/sm120a cubins without PTX and resolves against the
image's CUDA 12.8 runtime libraries. The driver version shown by `nvidia-smi` is
informational because it describes host compatibility rather than the container
toolkit.

Use a fresh environment containing this exact checkout and its pinned FlashInfer:

```bash
# Kernel, SWA-boundary, FP8-KV, and 128K decode/extend tests.
PROFILE=kernel PYTHON=python \
  scripts/attention_sink/run_hardware_validation.sh

# Actual checkpoint under Triton and FlashInfer.
PROFILE=server MODEL=/models/yccchen-olmo3-deploy TP=1 \
  PYTHON=python scripts/attention_sink/run_hardware_validation.sh

# A -> B -> A FlashRL cycle.
PROFILE=rl MODEL=/checkpoints/a RELOAD_MODEL=/checkpoints/b TP=1 \
  PYTHON=python scripts/attention_sink/run_hardware_validation.sh
```

Set `TP` to 2, 4, or 8 for shard qualification. The result directory contains
GPU/runtime metadata, git status, checkpoint checksums, test output, server logs,
deterministic probes, one-second GPU-memory and per-process RSS telemetry, and
Triton-versus-FlashInfer comparisons for BF16 and FP8 E4M3 KV caches. Set
`KV_CACHE_DTYPES=auto` only for a faster diagnostic run; release qualification
must run both defaults.

The release checkpoint is `chankhavu/yccchen-olmo3-deploy` at revision
`39beac79e6857df6d8a0dc27210f5affa4031c92`. It is a 32.5B BF16 model, so use
TP=2 on 80 GB H100 and qualify both TP=1 and TP=2 on B200. Download it once to
shared storage and create a copy-on-write sink variant:

```bash
hf download chankhavu/yccchen-olmo3-deploy \
  --revision 39beac79e6857df6d8a0dc27210f5affa4031c92 \
  --local-dir /models/yccchen-a
python scripts/attention_sink/make_sink_variant.py \
  /models/yccchen-a /models/yccchen-sink8 --value 8
```

The variant command requires reflink support and fails instead of making an
unplanned 65 GB copy. Its default ramp pattern makes every layer and query head
distinct, which is more sensitive to dropped, repeated, or incorrectly sharded
sink weights than a constant value. For release E2E, run A -> sink8 -> A under
both backends, weight quantizations, and KV-cache dtypes:

Place both paths on the same reflink-capable XFS or Btrfs volume. Many NFS and
ext4 model volumes do not support reflinks; on those filesystems, provision room
for a second 65 GB checkpoint rather than weakening the command to a silent
full copy.

```bash
CUDA_VISIBLE_DEVICES=0,1 PROFILE=rl TP=2 SKIP_KERNEL_TESTS=1 \
  MODEL=/models/yccchen-a RELOAD_MODEL=/models/yccchen-sink8 \
  QUANTIZATIONS="none fp8" KV_CACHE_DTYPES="auto fp8_e4m3" \
  PROBE_LENGTHS=128,4095,4096,4097,16384,131072 \
  CONTEXT_LEN=131328 MEMFRAC=0.70 \
  RESULTS=/workspace/results/h100-e2e \
  scripts/attention_sink/run_hardware_validation.sh
```

On B200, run the same matrix first with `CUDA_VISIBLE_DEVICES=0 TP=1`, then with
`CUDA_VISIBLE_DEVICES=0,1 TP=2`; use separate result directories. TP=1 verifies
the unsharded sink path, while TP=2 verifies rank-distinct sink sharding.

Separately validate the live tensor-transfer path for each backend. This sends
global sentinel values through OLMo's TP loader, checks every local sink shard,
proves behavior changes, restores the full checkpoint through FlashRL, and
proves A parity:

```bash
CUDA_VISIBLE_DEVICES=0,1 python \
  scripts/attention_sink/validate_live_sink_update.py \
  --model /models/yccchen-a --tp 2 --attention-backend flashinfer \
  --quantization fp8 --kv-cache-dtype fp8_e4m3 \
  --lengths 128,4097,16384 \
  --output /workspace/results/h100-live-flashinfer.json
```

The sink-only mutation sends a global 40-head tensor through OLMo's normal model
loader, which slices a distinct range on every TP rank. It does not use FlashRL,
because the transactional FlashRL loader requires a complete 771-weight
checkpoint. The FP8 A -> B -> A cycle above is the full-checkpoint FlashRL test.

Recommended hardware allocation:

| Qualification | H100 | B200 |
|---|---:|---:|
| Targeted unit/kernel suite | 1 GPU | 1 GPU |
| Real 32B server and A -> B -> A | 2 GPUs, TP=2 | 1 GPU TP=1, then 2 GPUs TP=2 |
| Synthetic sender-to-TP2 full BF16 update | Not recommended at 80 GB | 3 GPUs minimum |

The last row needs one sender/staging GPU plus two rollout GPUs concurrently.
The current full-weight receiver materializes the incoming 65 GB BF16 tensor set
before commit, so two 80 GB H100s cannot honestly qualify that production shape.
The disk-reload and sink-only tensor tests do not substitute for this final OPD
integration test. A real trainer may need substantially more than one sender GPU
for parameters, gradients, and optimizer state; three B200s is only the minimum
synthetic transfer topology, not a full training allocation.

Acceptance criteria:

- No FP8 test skips on H100 (`sm90`) or B200 (`sm100`).
- Kernel comparisons pass at declared BF16/FP8 tolerances.
- Sentinel sinks `-20`, `0`, and `+8` materially change output as eager predicts.
- Triton and FlashInfer produce identical greedy server tokens.
- Output-token logprobs differ by at most `0.05`.
- A -> B -> A reproduces A's output IDs and logprobs.
- Every TP rank reports 64 sink checksums, updated shards are rank-distinct, and
  restore reproduces the original per-engine sink checksum exactly.
- Cold head-dimension-128 FlashInfer JIT succeeds in an offline B200 container.

## Remaining Gaps

- Real `FlashInferAttnBackend` metadata and RadixAttention need full-server
  coverage; direct wrapper graph tests do not prove them.
- Full-server FP8 KV must pass separately on H100 and B200. FlashInfer 0.6.14 has
  no upstream sink+FP8 qualification.
- TP 4/8 and the production DP/PP topology still need the per-rank sink checksum
  test; the provided H100/B200 matrix covers TP 1/2.
- Non-unit FP8 K/V scales are not covered by the direct kernel fixture.
- Injected mid-commit GPU failure and single-TP-rank failure need fail-stop tests.
- Cold-start and FlashRL FP8 quantization of row-parallel projections should be
  aligned so the initial policy is independent of whether it came through reload.
- Concurrent generation must be aborted/retried during reload without admitting a
  mixed-version trajectory.
- The production image must install this fork, pass dependency checks, cold-build
  the JIT offline, and restart from a warm cache.
- Radix-cache update behavior is not qualified; OPD disables radix and flushes KV.
- Full-model 128K prefill is a separate expensive test. Kernel tests reach 128K
  decode and cached extend.
- Speculative decoding is deferred because Yi-Chia's sink-bearing DFlash draft is
  not compatible with SGLang's native DFlash model.

B200 is `sm100`, not `sm120`. Yi-Chia's `sm120` path requires a separate RTX
Blackwell system; neither H100 nor B200 validates it.
