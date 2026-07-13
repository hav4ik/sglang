# B200 Attention-Sink Container Decision

This record distinguishes host-driver compatibility from the CUDA libraries
inside a container. It applies to B200 rollout servers for the OLMo3
attention-sink work in this fork.

## Decision

Use `chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128` as the B200
qualification image. It contains CUDA toolkit/runtime 12.8, Torch
`2.11.0+cu128`, and FlashInfer 0.6.14, which is the exact FlashInfer version
targeted by the custom sink-kernel plumbing.

Here, `cu128` describes the operational compiler, runtime, framework, and
driver-API contract. It does not claim that every package version string is
`cu128`. Two audited exceptions remain: CUDA Python API wrappers at 12.9.4 and
the checksum-pinned `sglang-kernel==0.4.4+cu129` wheel. The latter contains
precompiled `sm90`/`sm100`/`sm120a` cubins without PTX and resolves CUDA 12
SONAMEs from the 12.8 image. It passed H100 testing but is not considered B200
qualified until the R570 hardware matrix passes.

The image does not use a normal dependency-resolving `pip install sglang` at
runtime. SGLang's public `torch==2.11.0` requirement accepts both `+cu128` and
`+cu130` local builds. The image therefore installs Torch, torchvision, and
torchaudio from PyTorch's cu128-only index with explicit `+cu128` requirements
before resolving other packages, and the editable bootstrap uses `--no-deps`.
The CUDA gate verifies the imported modules' local versions after every
bootstrap because PyTorch 2.11 distribution metadata omits the CUDA build tag.

Do not use `lmsysorg/sglang:v0.5.14-cu129` as the release image when the cluster
contract forbids operational CUDA components newer than 12.8. An R570 driver
can load many CUDA 12.9 applications through CUDA 12.x minor-version
compatibility, but that does not turn the container's CUDA 12.9 libraries into
CUDA 12.8.

## Official cu129 Image Audit

The Docker Hub artifact inspected on 2026-07-12 has these immutable identifiers:

| Item | Value |
|---|---|
| Multi-architecture index | `sha256:885da20811baa77e1c680cad0511d50efbaf94ebe43def9182bdf5290f47dbc5` |
| linux/amd64 manifest | `sha256:f54e858f2b51962cb6987a0b9c01ec16c7154692d4279475d7f2df412d0b1b8e` |
| linux/amd64 config | `sha256:ba1b822293c0520a6bac9f3c92d889ce82d6cc60381bf7d62b129e67c0c43548` |
| SGLang source | `49e384ce9d304648e9959666ecb8ce8cd98d0deb` |
| CUDA runtime/toolkit | 12.9.1 |
| Torch CUDA build | cu129 |
| FlashInfer | 0.6.12 |
| SGLang kernel | `0.4.4+cu129` |
| Compressed amd64 layers | approximately 17.46 GiB |

The official image source and this fork share merge base
`7e6587c94a1d0305815a14067c5d3cc02a9b0f36`, but they are not the same SGLang
release tree. At this documentation snapshot the sink fork has 763 commits not
in the image source, while the release source has eight commits not in the
fork. Installing this fork editable over the official v0.5.14 image would
therefore combine source and dependency sets that were not built together.

The image declares `CUDA_VERSION=12.9.1` and
`NVIDIA_REQUIRE_CUDA=cuda>=12.9` with an allowed R570 driver branch. Therefore:

- It is plausible as a separate R570 compatibility experiment.
- It is not compliant with this fork's strict CUDA 12.8 image gate.
- Its FlashInfer 0.6.12 installation and preloaded JIT artifacts do not match
  the qualified 0.6.14 sink integration.
- Its v0.5.14 Python/runtime dependency set is not assumed ABI-compatible with
  this fork's source tree.
- Deriving another image and upgrading FlashInfer would require deleting stale
  0.6.12 JIT artifacts and rerunning the complete kernel and server matrix.

NVIDIA documents CUDA 12.x minor-version compatibility for drivers from 525
through the range below 580, with feature restrictions. PTX JIT and features
that require a newer driver are important caveats. FlashInfer and Triton JIT
CUDA code at first use, so successful container launch alone is not release
evidence.

References:

- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
- [CUDA 12.9 release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
- [SGLang v0.5.14-cu129 image](https://hub.docker.com/layers/lmsysorg/sglang/v0.5.14-cu129/images/)

## B200 Bring-Up

Mount the JIT caches persistently. The first run compiles architecture-specific
FlashInfer and Triton kernels for `sm100`; later containers reuse them.

```bash
docker pull chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/cache:/cache" \
  -v "$PWD/workspace:/workspace" \
  -v /shared/models:/models \
  chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128

SGLANG_REF=codex/flashinfer-attention-sink sglang-sink-bootstrap
sglang-sink-check-cuda
python /workspace/sglang/scripts/attention_sink/check_b200_environment.py
sglang-sink-tests
```

The source command above works immediately after bootstrapping the currently
published image. Newly built images also install it as `sglang-sink-check-b200`.
The check requires an R570 driver by default, every visible GPU to report
compute capability 10.0, FlashInfer 0.6.14, and the complete CUDA 12.8 gate.
Override `--driver-major` only when intentionally qualifying a different driver
branch.

At runtime the CUDA gate checks both sides of the API boundary:

- `nvcc`, Torch, `libcudart`, toolkit packages, and filesystem toolkits must be
  CUDA 12.8.
- The injected `libcuda.so.1` Driver API must report 12.8 as well.
- `cuda-compat-12-9` and all `cuda-compat-13-*` packages are rejected.

NVIDIA's forward-compatibility mechanism can expose a newer `libcuda` interface
over an older kernel driver, and it is limited to data-center GPUs, selected
server-ready RTX systems, and Jetson. That is a valid deployment mechanism for
an intentionally newer CUDA application, but it is not used by this strict
cu128 image. R570 is the driver paired with CUDA 12.8, so B200 does not need a
CUDA 13 compatibility interface for this stack.

## Backend Fallback

The checkpoint is backend-neutral. Start either complete sink implementation
explicitly:

```bash
python -m sglang.launch_server --model-path /models/yccchen-a \
  --attention-backend flashinfer --page-size 1

python -m sglang.launch_server --model-path /models/yccchen-a \
  --attention-backend triton --page-size 1
```

Changing backend requires a server restart; it is not a runtime control-plane
operation. A restart ends active requests and discards that server's KV cache.
After startup, complete AsyncRL checkpoint updates, including all 64 sink
tensors, remain in-place and no-flush for the selected backend. Do not use the
prefill/decode split-backend flags for this model until that mixed combination
has its own qualification run.

Run the real model first on one B200, then TP=2. Keep Triton and FlashInfer
explicit; automatic backend selection is not qualified.

```bash
CUDA_VISIBLE_DEVICES=0 PROFILE=server TP=1 SKIP_KERNEL_TESTS=1 \
  MODEL=/models/yccchen-a BACKENDS="triton flashinfer" \
  QUANTIZATIONS="none fp8" KV_CACHE_DTYPES="auto fp8_e4m3" \
  PROBE_LENGTHS=128,4095,4096,4097,16384,131072 \
  CONTEXT_LEN=131328 MEMFRAC=0.65 \
  RESULTS=/workspace/results/b200-tp1 \
  scripts/attention_sink/run_hardware_validation.sh

CUDA_VISIBLE_DEVICES=0,1 PROFILE=rl TP=2 SKIP_KERNEL_TESTS=1 \
  MODEL=/models/yccchen-a RELOAD_MODEL=/models/yccchen-sink8 \
  BACKENDS="triton flashinfer" \
  QUANTIZATIONS="none fp8" KV_CACHE_DTYPES="auto fp8_e4m3" \
  PROBE_LENGTHS=128,4095,4096,4097,16384,131072 \
  CONTEXT_LEN=131328 MEMFRAC=0.65 \
  RESULTS=/workspace/results/b200-tp2-rl \
  scripts/attention_sink/run_hardware_validation.sh
```

FP8 KV remains a separate qualification item. A passing BF16-KV run does not
qualify FP8 KV, and direct-kernel parity does not replace the full server test.
The rollout image qualifies SGLang inference and AsyncRL weight reload; it is not
the OLMo-core optimizer/training container.

## Building

The reproducible strict image is built by:

```bash
IMAGE=chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128 \
  scripts/attention_sink/build_push_cu128.sh
```

The build script requires a clean committed checkout and at least 70 GiB free,
uses an isolated BuildKit builder, and pushes both the moving tag and an
immutable source-revision tag. Do not bypass the disk gate on a nearly full
host. Remove only an explicitly reviewed isolated builder cache, or run the
build on a clean machine with sufficient storage.
