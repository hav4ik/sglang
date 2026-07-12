# OLMo3 Attention-Sink Audit and Qualification Record

This document is the engineering record for serving
`chankhavu/yccchen-olmo3-deploy` with learned attention sinks in this SGLang
fork. It records what changed, what was actually tested, known numerical and
operational differences, and the remaining release gates. Operational commands
live in [attention_sinks.md](attention_sinks.md).

## Snapshot

| Item | Audited value |
|---|---|
| SGLang branch | `codex/flashinfer-attention-sink` |
| Documentation snapshot | `1b510884c` and descendants |
| Original image checkout | `4df5f5431d83f3bc4da366cca41ffce8496fef4a` |
| Model | `chankhavu/yccchen-olmo3-deploy` |
| Model revision | `39beac79e6857df6d8a0dc27210f5affa4031c92` |
| Yi-Chia source reference | `bc03a2c71a076990deaad3d712c6889682e12c69` |
| CUDA contract | Toolkit, runtime, nvcc, and Torch CUDA at 12.8 |
| Torch | `2.11.0+cu128` |
| FlashInfer | `0.6.14` |
| Primary tested hardware | 2x H100 80 GB, TP=2 |

The image permits the checksum-pinned `sglang-kernel==0.4.4+cu129` wheel as a
documented native-wheel exception. It contains architecture cubins and runs
against the CUDA 12.8 runtime. CUDA Python API packages at 12.9.4 are bindings,
not a newer toolkit or runtime.

## Status

The implementation is suitable for a controlled H100 canary with explicit
Triton or FlashInfer, FP8 or BF16 weights, BF16 KV, page size 1, radix disabled,
and no speculative decoding. It is not a general release qualification.

| Area | Status | Evidence or blocker |
|---|---|---|
| Sink checkpoint population | Verified | 64 tensors loaded and exact per-rank checksums observed |
| Triton prefill/decode sink math | Verified | Direct eager comparison and captured production replay |
| FlashInfer prefill/decode sink math | Verified for BF16 KV | Dedicated sink wrapper, eager comparison, and production replay |
| OLMo 4096-token SWA convention | Verified after fix | 4095 predecessors plus current token; sink consumes no position |
| CUDA graph sink reload | Verified directly | Captured graph reads mutated sink storage without pointer replacement |
| TP2 disk A -> B -> A | Verified on H100 | Exact sink restore, changed B behavior, restored A outputs |
| In-flight no-flush disk reload | Verified on H100 | Active request retained KV and completed after resume |
| Full-model Triton/FlashInfer numerical parity | Failing diagnostic gate | Maximum observed delta exceeds `0.05` |
| Independent Yi-Chia eager comparison | Diagnostic only | One prompt; no predeclared release threshold |
| Patched FA3 training reference | Not available in rollout image | Requires a separate SM90 native build |
| FlashInfer FP8 KV | Sink math verified; policy quality unqualified | Current prefill chunk is quantized before attention |
| Real-model 128K prefill | Not tested | Synthetic kernels reached 128K; server probes reached 16K |
| B200 TP1/TP2 | Not tested | Requires separate hardware qualification |
| sm120 | Not tested | B200 is sm100 and does not qualify sm120 |
| DFlash/speculative | Deferred | Yi-Chia's sink-bearing draft is not SGLang's native draft |

## Implemented Changes

### Model and checkpoint contract

- Detect `Olmo3SinkForCausalLM`, `model_type=olmo3_sink`, and sink-bearing OLMo
  configuration.
- Create one FP32 runtime sink parameter per local query head. Checkpoint BF16
  values promote exactly to FP32.
- Slice the global 40-head sink tensor contiguously by TP rank.
- Reject missing sink tensors at initial load.
- Require the complete 771-weight OLMo checkpoint for transactional FlashRL
  reloads.
- Preserve sink parameter storage addresses across reload so CUDA graphs continue
  reading the same pointer.
- Add sink-only runtime checksums and checkpoint inspection/variant tools.

### Attention backends

- Keep Triton as an explicit rollout option and route sinks through its existing
  extend and split-KV decode kernels.
- Add FlashInfer's dedicated paged attention-sink wrapper for prefill, cached
  extend, and decode. Ordinary paged wrappers are not used for sink models.
- Build a dtype-complete FlashInfer JIT identity so BF16-KV and FP8-KV sink
  kernels cannot collide in the cache.
- Force paged prefill for sink models and reject unsupported FlashInfer paths:
  page size greater than 1, DCP, MIS, custom masks, and nonzero soft-cap.
- Pass causal and window metadata consistently during normal execution and CUDA
  graph planning.

### Sliding-window correction

The model's `sliding_window=4096` means 4096 real KV positions, including the
current token. SGLang stores `window_left`, so OLMo converts this to 4095
predecessors. Triton decode metadata originally retained only 4095 total keys.
The production buffer, CUDA graph allocation, and tests now retain
`window_left + 1` real keys. The virtual sink is denominator-only and never
occupies a token, position, RoPE entry, K/V row, or cache slot.

### AsyncRL/OPD reload

The qualified OPD v2 path is a complete checkpoint update from disk:

```text
save and validate checkpoint
-> pause generation in-place
-> load/quantize/validate all weights
-> copy into graph-stable runtime storage
-> keep active-request KV
-> resume generation
```

The test cycle is cold A0 -> warm reload A1 -> sink-mutated B -> restored A2.
A1 and A2 use the same FlashRL global-before-TP quantization path. Cold A0 can
differ because startup quantizes TP-local rows. Before serving RL traffic, warm
reload the initial checkpoint so all policy versions use the reload path.

Sink-only tensor transfer is a separate TP-loader diagnostic and runs without
FlashRL quantization. A partial sink update against FlashRL is correctly rejected
because the transactional loader requires all 771 tensors.

No-flush reload deliberately permits an active trajectory to contain old-weight
KV followed by new-weight decode. It does not provide per-request policy-version
isolation. Shared radix prefixes are disabled because otherwise a new request
could reuse old-policy KV after an update.

## Attention-Sink Mathematics

For query head `h`, token logits `z_i`, value vectors `v_i`, and sink logit `s_h`:

```text
D_h = exp(s_h) + sum_i exp(z_i)
O_h = sum_i exp(z_i) v_i / D_h
```

The sink contributes zero to the value numerator. Yi-Chia eager, patched FA3,
SGLang Triton, and the dedicated FlashInfer wrapper implement this definition.
They are algebraically equivalent but not bitwise equivalent because they use
different tiling, reduction orders, exponential implementations, and output
rounding.

Yi-Chia's eager reference concatenates the sink and real logits before FP32
softmax. Patched FA3 and SGLang Triton add `exp(sink - M)` to the accumulated
denominator after the real-token maximum `M` is known. That difference is not a
Triton deviation from patched FA3.

## FP8 Interpretation

### FP8 weights are not FP8 KV

`--quantization fp8` quantizes model matrices. With `--kv-cache-dtype auto`, the
KV cache remains BF16. Sinks remain FP32. The observed `0.217067` H100 backend
delta used FP8 weights with BF16 KV and therefore is unrelated to the FP8-KV
current-chunk policy.

`--kv-cache-dtype fp8_e4m3` quantizes stored K/V. It does not quantize the sink.

### Backend current-chunk policy

| Path | Cached prefix | Current prefill/extend chunk |
|---|---|---|
| Yi-Chia patched FA3 training | n/a | BF16 |
| Yi-Chia custom Triton rollout | dequantized FP8 | BF16 |
| SGLang Triton | dequantized FP8 | BF16 |
| SGLang FlashInfer paged | dequantized FP8 | dequantized FP8 |
| SGLang paged FA3 | dequantized FP8 | dequantized FP8 |

FlashInfer's paged prefill writes the current K/V into the FP8 cache before
calling attention. Triton clones and quantizes the cache copy while preserving
the current chunk in BF16 for that forward. This behavior predates the sink work;
the sink branch inherited it when enabling FlashInfer's dedicated paged wrapper.

FlashInfer still computes the correct sink formula for the FP8 K/V it receives.
Quantized K changes real-token logits and therefore changes the expected sink
mass:

```text
p_sink = exp(s) / (exp(s) + sum_i exp(q K_i / sqrt(d)))
```

Quantized V changes the numerator. Neither effect means that the sink was
dropped, duplicated, or quantized. The unresolved question is model and RL-policy
quality relative to BF16-current Yi-Chia/Triton execution.

Existing direct FP8 tests intentionally pre-quantize all K/V so both kernels see
identical inputs. They validate sink arithmetic and K/V scales but cannot validate
the production current-chunk policy. A production-backend test must start with a
BF16 current chunk, let each backend write its own cache, and compare against the
chosen policy oracle.

## Observed H100 Evidence

### Checkpoint

The downloaded checkpoint contained 771 indexed tensors and 64 sinks. Observed
sink statistics were:

```text
minimum  -0.30859375
maximum  13.6875
mean      6.636940765380859
SHA-256   e8c24de77d052964e08375170a77edccaf4cf6805e62257d4d46c7e1a4d04fc8
```

The adversarial ramp variant changed every layer/head sink and produced a
different exact checksum.

### Kernel and trace results

- Registered direct sink tests passed on H100 for BF16/FP8 K/V, full/SWA,
  prefill/cached-extend/decode, target 40:8 GQA and TP2-local 20:4 shapes.
- Long synthetic decode and cached-extend tests passed at 4095, 4096, 4097,
  32768, and 131072 tokens where selected.
- Captured layer-0 Q/K/V/sinks were identical between BF16-KV backend runs.
- Layer-0 maximum errors against FP32 eager were approximately `0.00975` for
  Triton and `0.00780` for FlashInfer; backend maximum difference was `0.015625`.
- Representative prefill replays passed for layers 0, 3, 31, and 63 on both TP
  ranks.
- Representative decode replays passed for layers 0, 3, 31, 62, and 63 on both
  TP ranks.

Trace replay validates a selected captured layer. All-layer printed deltas are
diagnostic because backend states naturally diverge after earlier layers. FP8-KV
traces currently lack cache scales and page metadata and must not be treated as
faithful cache replays.

### Full-model divergence

Greedy output tokens matched in the reported probes, but the fixed `0.05`
Triton/FlashInfer logprob gate failed:

| Weights | KV | Maximum observed backend delta |
|---|---|---:|
| FlashRL FP8 | BF16 (`auto`) | `0.217067` at length 128 |
| BF16 | BF16 (`auto`) | `0.110017` at length 128 |

Do not increase the threshold to fit these observations. This is a model-level
64-layer difference, not a per-layer sink-kernel error.

Yi-Chia eager at length 128 produced tokens `[42, 42, 42, 42]`. Both SGLang
backends produced the same tokens. Maximum selected-token logprob error was
`0.166686` for Triton and `0.109654` for FlashInfer on that one prompt. This makes
FlashInfer closer for that sample, not generally proven superior. The comparator
was diagnostic and the eager run used a different non-TP execution topology.

## Adversarial Audit Findings

Four audit scopes were used: backend mathematics/plumbing, AsyncRL transfer,
test-harness validity, and deployment/hardware compatibility. The actionable
findings are:

1. Production FP8-KV prefill has the backend policy difference documented above;
   synthetic tests hide it.
2. The independent eager comparator reports errors but does not enforce a
   predeclared numerical threshold or reject different-token comparisons.
3. Patched FA3 is absent from the rollout container, so training-kernel parity is
   not yet measured.
4. Warm A1/restored A2 compare exact sinks but not complete per-rank runtime
   weights and FP8 scales.
5. Trace comparison gates selected replay layers; printed all-layer differences
   are not gates.
6. Trace and result artifacts lack a mandatory run ID and complete provenance;
   stale files can contaminate reused result directories.
7. The long-context suite needs a sink-on/sink-disabled control strong enough to
   fail if long-context kernels ignore the sink.
8. Production FlashInfer piecewise CUDA-graph extend needs a sink-enabled,
   page-size-1, nonempty-SWA-prefix regression test.
9. Calibrated FP8-KV checkpoints need OLMo remapping from
   `self_attn.{k,v}_scale` to `self_attn.attn.{k,v}_scale`.
10. Automatic hardware backend selection can choose unqualified FA3 or TRT-LLM
    MHA paths; sink deployments must select an audited backend explicitly.
11. CUDA 12.8 validation records but does not require H100/B200 architecture, and
    pytest skips are not currently promoted to qualification failures.
12. TP shape tests do not substitute for real distributed TP2/4/8, multi-node,
    DP, or PP execution.
13. Full-model prompts are narrow: repeated token 42, four generated tokens, and
    selected-token logprobs rather than an independently declared distribution
    metric.
14. Repeated reload stress has not established a stable GPU/host memory plateau.
15. Full sender-to-rollout weight transfer is not qualified by disk reload; OPD
    v2 currently uses disk, but any future distributed/tensor protocol needs its
    own complete-checkpoint test.

These findings do not invalidate the verified BF16-KV sink arithmetic. They
limit the supported deployment envelope and identify where the qualification
harness can otherwise produce false confidence.

## Release Test Plan

### P0: correctness and policy fidelity

1. Build a separate H100 patched-FA3 reference image. Verify the operator schema
   contains the trailing `Tensor? sink` argument; an ordinary FA3 package is not
   sufficient.
2. Declare a prompt corpus and tolerances before execution. Compare patched FA3,
   Yi-Chia eager, Triton, and FlashInfer using identical token IDs or stop at the
   first token divergence.
3. Add production-path BF16 and FP8-KV first-prefill/cached-extend tests. Include
   empty and nonempty prefixes and 4095/4096/4097 window boundaries.
4. Compute independent expected per-rank sink hashes for A and B. Verify runtime
   hashes equal the expected values, not merely that A and B differ.
5. Compare complete runtime weight/scale checksums for warm A1 and restored A2.
6. Either keep radix disabled as a hard AsyncRL invariant or implement and test
   versioned/shared-prefix invalidation while retaining active-request KV.

### P1: hardware and execution matrix

1. H100 TP2 and B200 TP1/TP2 with explicit Triton and FlashInfer.
2. BF16 and FP8 weights crossed with BF16 and FP8 KV.
3. CUDA graph on/off, first prefill, chunked extend, decode, and mixed batch
   lengths.
4. Full-model 128K prefill and decode, with memory telemetry and no skipped FP8
   cases.
5. Sink-enabled piecewise prefill graph with a nonempty SWA prefix.
6. Ten or more A/B reload cycles with exact checksums and memory plateau checks.
7. Load a calibrated FP8-KV checkpoint and validate nonunit K/V scales end to end.

### P2: broader compatibility

- Explicit startup rejection for unqualified sink/backend combinations.
- TP4/TP8, multi-node TP, production DP/PP, and rank-failure behavior.
- DFlash draft loading, draft sinks, target verification, quantized draft, and
  SWA eviction after the lower-priority speculative work is resumed.
- Automatic backend selection only after FA3 and TRT-LLM MHA sink qualification.

## Release Criteria

A result is release evidence only when it records an immutable SGLang commit,
model revision, backend, weight/KV dtype, graph mode, TP topology, GPU model and
compute capability, CUDA versions, and a fresh run ID. Required matrix entries
must fail on skipped tests or missing artifacts.

The release gate requires:

- Correct greedy tokens and finite declared distribution metrics against an
  independent reference.
- Sink-on controls that would fail for dropped, repeated, constant, or incorrectly
  sharded sinks.
- Production-path FP8-KV tests, not only already-quantized direct-kernel inputs.
- Exact expected sink hashes on every TP rank.
- Exact complete A1/A2 runtime checksums after transactional reload.
- Successful active-request completion across no-flush reload with radix disabled
  or version-safe.
- Real-model 128K and required H100/B200 matrices.
- No post-result relaxation of thresholds.

Until these criteria pass, use the controlled H100 canary envelope and treat
FlashInfer FP8 KV, B200, 128K full-model execution, automatic backend selection,
and speculative decoding as unqualified.
