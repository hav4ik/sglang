# OLMo3 Attention Sinks

This branch adds inference and RL reload support for OLMo3 models with one learned
attention-sink logit per query head. The target artifact is
`chankhavu/yccchen-olmo3-deploy` (`Olmo3SinkForCausalLM`). Yi-Chia Chen's Triton
path remains the rollout default; FlashInfer is an explicit, release-gated option.
The audited model revision is `39beac79e6857df6d8a0dc27210f5affa4031c92`.
The complete implementation and qualification record is maintained in
[attention_sinks_audit.md](attention_sinks_audit.md). The B200 image choice,
CUDA compatibility analysis, and bring-up commands are recorded in
[attention_sinks_b200_container.md](attention_sinks_b200_container.md).

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

The sink is not a sequence position and does not consume the sliding window.
OLMo training converts `sliding_window=4096` to FlashAttention's inclusive
`window_size=(4095, 0)`, which retains 4096 real KV positions. The sink is added
after that mask as one extra denominator-only entry, so a saturated SWA query
normalizes over 4096 real-token logits plus the sink logit.

FlashInfer 0.6.14's dedicated wrapper gives BF16-KV and FP8-KV sink kernels the
same JIT cache identity because that identity omits the KV dtype. SGLang uses
FlashInfer's own dtype-complete sink URI helper when constructing the otherwise
identical custom kernel, preventing a shared BF16 cache entry from being reused
as FP8 (or vice versa).

### FP8 weights and FP8 KV are independent

Two unrelated command-line settings contain the term "FP8":

| Setting | What is quantized | Sink implication |
|---|---|---|
| `--quantization fp8` | Linear model weights through FlashRL | Sinks stay FP32 at runtime; KV stays BF16 when `--kv-cache-dtype auto` |
| `--kv-cache-dtype fp8_e4m3` | Stored attention K/V tensors | Sinks still stay FP32, but token logits and value mixtures use quantized K/V |

Consequently, an `fp8/auto` result exercises FP8 weights with BF16 KV and does
not exercise the FP8-KV behavior described below. The observed H100 full-model
backend deltas of `0.217067` (`fp8/auto`) and `0.110017` (`none/auto`) are not
caused by FP8 KV.

### FP8-KV current-chunk policy

The sink equation itself is unchanged under FP8 KV. FlashInfer receives the
same per-query-head FP32 sink and correctly adds it to the denominator for the
K/V tensors supplied to its dedicated sink kernel. The backend difference is
which representation of the current prefill/extend chunk is supplied:

| Execution path | Cached prefix | Current prefill/extend chunk |
|---|---|---|
| Yi-Chia patched FA3 training | n/a | BF16 |
| Yi-Chia custom Triton rollout | dequantized FP8 | BF16 |
| SGLang Triton | dequantized FP8 | BF16 |
| SGLang FlashInfer paged | dequantized FP8 | dequantized FP8 |
| SGLang paged FA3 | dequantized FP8 | dequantized FP8 |

For a query `q`, FP32 sink `s`, an FP8 quantizer `Q`, and attention function
`A`, an initial no-prefix prefill therefore behaves approximately as:

```text
Triton:    A(q, K_bf16, V_bf16, s)
FlashInfer A(q, dequantize(Q(K_bf16)), dequantize(Q(V_bf16)), s)
```

For cached extend, both use dequantized FP8 prefix K/V, while only Triton keeps
the new chunk in BF16 for that forward. Once a token has entered the cache,
decode reads FP8 K/V in both paths. This is an existing paged-backend precision
policy, not behavior introduced by the attention-sink formula. This branch
inherited it when it enabled the dedicated FlashInfer sink wrapper.

Quantizing K changes real-token logits and therefore can change the resulting
sink probability:

```text
p_sink = exp(s) / (exp(s) + sum_i exp(q K_i / sqrt(d)))
```

That is the mathematically expected consequence of quantized K, not evidence
that the sink was dropped or quantized. Quantizing V changes the numerator. The
direct kernel tests prove that FlashInfer applies the correct sink equation for
its FP8 inputs; they do not prove equivalence to Triton's mixed FP8-prefix / BF16
current-chunk policy because those tests intentionally provide both backends the same
already-quantized K/V.

FP8 KV is therefore a quality and policy-fidelity qualification item rather than
a known sink-kernel correctness defect. It is acceptable if FlashInfer's normal
FP8-KV semantics are the desired serving contract and model-level quality is
gated independently. It is not yet qualified as a faithful reproduction of
Yi-Chia's training or custom Triton rollout numerics.

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
| FP8 E4M3 KV | Mixed FP8 prefix / BF16 current chunk | FP8 prefix and current chunk; model-quality release gate |
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

AsyncRL/OPD uses this replica barrier:

```text
save and validate checkpoint
-> pause scheduler forwards in-place on every replica
-> reload every replica without flushing active-request KV
-> verify every response
-> resume every replica
-> advance the global version
```

Active requests can therefore span a weight boundary: their existing KV was
produced by the old weights and their resumed decode uses the new weights. If any
stage fails, the global version does not advance and paused replicas remain
paused. Per-request policy-version isolation requires versioned weights and KV;
the in-place protocol intentionally does not provide it.

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

The serving backend is selected at process startup with
`--attention-backend triton` or `--attention-backend flashinfer`. Both consume
the same checkpoint and support complete AsyncRL sink reloads. Switching
backends requires a server restart and therefore does not preserve active
requests or KV cache; split prefill/decode backend combinations are not in the
qualified envelope.

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
python /workspace/sglang/scripts/attention_sink/check_b200_environment.py  # B200/R570 only
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

Reload validation needs transient device memory beyond the static model and KV
allocations. On 80 GB GPUs, lower `MEMFRAC` if reload runs out of memory. For
batch-one smoke tests, set `CUDA_GRAPH_MAX_BS_DECODE=1` and
`CUDA_GRAPH_MAX_BS_PREFILL=1`; this still captures and replays both graph paths
without reserving graph pools for batch sizes the smoke test never sends.

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
  --quantization none --kv-cache-dtype auto \
  --lengths 128,4097,16384 \
  --output /workspace/results/h100-live-flashinfer.json
```

The sink-only mutation sends a global 40-head tensor through OLMo's normal model
loader, which slices a distinct range on every TP rank. It does not use FlashRL,
because the transactional FlashRL loader requires a complete 771-weight
checkpoint. Run this diagnostic with `--quantization none`; a sink-only tensor
update against a FlashRL model is intentionally rejected as an incomplete
checkpoint. The FP8 A -> B -> A disk cycle above is the full-checkpoint FlashRL
test and the relevant OPD v2 path.

### Backend divergence trace

The fixed full-model backend gate is currently failing on H100 even though
greedy tokens, reload parity, sink checksums, and direct kernel tests pass. With
FP8 weights and BF16 KV, the largest observed output-token logprob delta is
`0.217067` at prompt length 128. The BF16-weight control reaches `0.110017`, so
the discrepancy is not isolated to FP8 weight conversion. Do not raise the
`0.05` gate based on these observations.

An audit also found that Triton decode retained only 4095 keys for OLMo's
4096-token SWA window. Production metadata, CUDA-graph buffers, and the backend
reference test now consistently retain the current token plus 4095 predecessors.
That bug affects boundary and long-context probes but cannot explain the
length-128 discrepancy, which is why the layer trace remains necessary.

Capture the actual 128-token attention tensors from each backend using separate
result and trace directories:

```bash
rm -rf /workspace/results/sink-trace-{triton,flashinfer}

CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_ATTENTION_SINK_TRACE_DIR=/workspace/results/sink-trace-triton/tensors \
SGLANG_ATTENTION_SINK_TRACE_TOKENS=128 \
PROFILE=server TP=2 SKIP_KERNEL_TESTS=1 \
MODEL=/models/yccchen-a BACKENDS=triton \
QUANTIZATIONS=fp8 KV_CACHE_DTYPES=auto PROBE_LENGTHS=128 \
PROBE_OUTPUT_TOKENS=1 CONTEXT_LEN=8192 MEMFRAC=0.55 \
DISABLE_CUDA_GRAPH=1 \
RESULTS=/workspace/results/sink-trace-triton \
scripts/attention_sink/run_hardware_validation.sh

CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_ATTENTION_SINK_TRACE_DIR=/workspace/results/sink-trace-flashinfer/tensors \
SGLANG_ATTENTION_SINK_TRACE_TOKENS=128 \
PROFILE=server TP=2 SKIP_KERNEL_TESTS=1 \
MODEL=/models/yccchen-a BACKENDS=flashinfer \
QUANTIZATIONS=fp8 KV_CACHE_DTYPES=auto PROBE_LENGTHS=128 \
PROBE_OUTPUT_TOKENS=1 CONTEXT_LEN=8192 MEMFRAC=0.55 \
DISABLE_CUDA_GRAPH=1 \
RESULTS=/workspace/results/sink-trace-flashinfer \
scripts/attention_sink/run_hardware_validation.sh

CUDA_VISIBLE_DEVICES=0 python \
  scripts/attention_sink/compare_attention_traces.py \
  /workspace/results/sink-trace-triton/tensors \
  /workspace/results/sink-trace-flashinfer/tensors \
  --tp-rank 0 --replay-layer 0 \
  | tee /workspace/results/sink-trace-rank0.json
```

Repeat the comparison with `--tp-rank 1`. The trace records Q, K, V, sinks, and
the attention output for all 64 layers, about 200 MB per backend at TP=2. The
comparison identifies where backend inputs first diverge and replays layer 0
through eager, Triton, and FlashInfer using identical checkpoint activations.
Tracing is disabled unless `SGLANG_ATTENTION_SINK_TRACE_DIR` is set. The trace
run disables CUDA graphs so the real request executes the instrumented Python
forward path; CUDA-graph behavior remains covered by the normal qualification.

Reload probes use `pause_generation(mode="in_place")`, update with
`flush_cache=False`, and then continue generation. This is the AsyncRL hot-swap
contract: scheduler forwards are quiesced during mutation, and active-request KV
is retained. An active request can therefore resume with KV produced by the old
weights and generate later tokens with the new weights. This is not per-request
model-version isolation. The validation server disables radix caching so a new
request cannot reuse another request's old-version prefix KV; deployments that
retain shared prefix caches need model-versioned cache entries or an explicit
invalidation policy.

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

### Current deployment envelope

The evidence collected so far supports a controlled H100 canary with this exact
configuration:

```text
hardware             H100, TP=2
attention backend    explicitly triton (default canary) or flashinfer
weight dtype         BF16 or FlashRL FP8
KV cache dtype       BF16 / auto
page size            1
radix cache          disabled
DCP / MIS / spec     disabled
reload source        complete checkpoint from disk
reload cache policy  flush_cache=false, active-request KV intentionally retained
tested model context through 16384 tokens
```

For FlashRL, perform a same-checkpoint warm reload before accepting rollouts so
cold startup and later policy versions use the same global-before-TP quantization
path. Do not treat this envelope as qualification for B200, sm120, full-model
128K, explicit FP8 KV, automatic backend selection, shared radix prefixes, or
speculative decoding.

### Adversarial audit findings

Independent static audits found no unconditional sink drop in the ordinary
Triton or dedicated FlashInfer paged paths. They did identify these release and
test-validity gaps:

1. Production FP8-KV prefill uses the backend-specific current-chunk policies
   documented above. Existing synthetic tests pre-quantize all K/V and cannot
   detect this distinction.
2. The Yi-Chia eager comparator is diagnostic: it reports errors but has no
   predeclared numerical acceptance threshold. The patched FA3 training kernel is
   not installed in the rollout image.
3. Full-model A -> B -> A verifies sink checksums exactly but does not yet compare
   complete runtime weight and FP8-scale checksums between warm A1 and restored A2.
4. A sink-only live tensor update cannot pass through the transactional FlashRL
   loader, which correctly requires all 771 checkpoint weights. Sink-only TP
   slicing and full FP8 reload are separate tests.
5. FP8 trace files do not capture cache dtype, K/V scales, page tables, or prefix
   metadata. They must not be interpreted as FP8-cache replays.
6. Trace comparison gates the selected replay layer, while the all-layer deltas
   are diagnostic. Qualification must replay a predeclared layer set on every TP
   rank rather than relying on the printed all-layer table.
7. Result and trace directories need fresh-run provenance. Reusing a directory
   can mix artifacts from different commits or backend configurations.
8. Calibrated FP8-KV checkpoints may contain `self_attn.{k,v}_scale`; OLMo still
   needs the standard remapping to RadixAttention's
   `self_attn.attn.{k,v}_scale` before those artifacts are qualified.
9. No-flush reload with shared radix prefixes admits old-policy KV reuse by new
   requests. Current AsyncRL qualification disables radix; production must keep
   it disabled or implement versioned/invalidation semantics.
10. Automatic H100/B200 backend selection includes paths outside the explicit
    Triton/FlashInfer matrix. Sink deployments must select a qualified backend
    explicitly until FA3 and TRT-LLM MHA are separately tested.

The audit changes the confidence classification, not the sink equation: BF16-KV
kernel and trace results remain valid, while FP8-KV and broad deployment claims
need the additional gates below.

Release acceptance criteria (the current canary does not yet satisfy every row):

- No FP8 test skips on H100 (`sm90`) or B200 (`sm100`).
- Kernel comparisons pass at declared BF16/FP8 tolerances.
- Production FP8-KV prefill and cached extend are measured using each backend's
  actual cache-write path; synthetic already-quantized K/V is not sufficient.
- Sentinel sinks `-20`, `0`, and `+8` materially change output as eager predicts.
- Triton and FlashInfer produce identical greedy server tokens.
- Output-token logprobs differ by at most `0.05`.
- A -> B -> A reproduces A's output IDs and logprobs.
- Warm A1 and restored A2 have identical complete per-rank runtime checksums,
  including FP8 weight scales, not only identical sink checksums.
- Every TP rank reports 64 sink checksums, updated shards are rank-distinct, and
  restore reproduces the original per-engine sink checksum exactly.
- A request observed as running before an in-place update completes all requested
  decode tokens after the no-flush update resumes generation.
- Cold head-dimension-128 FlashInfer JIT succeeds in an offline B200 container.
- Independent patched-FA3 and eager-reference thresholds are declared before
  running the release prompt corpus; reference comparison exits nonzero on a
  failed threshold or token mismatch.
- Every result manifest records an immutable SGLang commit, model revision,
  backend, weight/KV dtype, graph mode, TP topology, GPU architecture, and run ID.

## Remaining Gaps

- Real `FlashInferAttnBackend` metadata and RadixAttention need full-server
  coverage; direct wrapper graph tests do not prove them.
- Full-server FP8 KV must pass separately on H100 and B200. FlashInfer 0.6.14 has
  no upstream sink+FP8 qualification, and its current prefill chunk is quantized
  before attention rather than retained in BF16 like Triton.
- TP 4/8 and the production DP/PP topology still need the per-rank sink checksum
  test; the provided H100/B200 matrix covers TP 1/2.
- Non-unit FP8 K/V scales are covered directly; their full-server FP8-E4M3 path
  still needs H100 and B200 qualification.
- Injected mid-commit GPU failure and single-TP-rank failure need fail-stop tests.
- Cold-start and FlashRL FP8 quantization of row-parallel projections should be
  aligned so the initial policy is independent of whether it came through reload.
- In-place, no-flush reload intentionally admits mixed-version trajectories for
  active requests. The integration test proves request survival with radix
  disabled; shared radix KV is not yet versioned across policy updates.
- The production image must install this fork, pass dependency checks, cold-build
  the JIT offline, and restart from a warm cache.
- Radix-cache update behavior is not qualified; current AsyncRL validation disables
  radix while preserving active-request KV.
- Full-model 128K prefill is a separate expensive test. Kernel tests reach 128K
  decode and cached extend.
- Speculative decoding is deferred because Yi-Chia's sink-bearing DFlash draft is
  not compatible with SGLang's native DFlash model.

B200 is `sm100`, not `sm120`. Yi-Chia's `sm120` path requires a separate RTX
Blackwell system; neither H100 nor B200 validates it.
