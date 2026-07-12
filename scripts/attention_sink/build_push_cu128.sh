#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-chankhavu/proofpilot-sglang-sink:flashinfer-sink-cu128}"
BUILDER="${BUILDER:-sglang-sink-cu128-$$}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
REVISION="$(git -C "$ROOT" rev-parse HEAD)"
REVISION_TAG="${IMAGE%:*}:cu128-${REVISION:0:12}"

free_kb="$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')"
if [ "$free_kb" -lt "$((MIN_FREE_GB * 1024 * 1024))" ]; then
  echo "ERROR: need at least ${MIN_FREE_GB} GiB free; refusing Docker build" >&2
  exit 1
fi
if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "ERROR: commit the candidate before building a release image" >&2
  exit 1
fi

cleanup() {
  docker buildx rm -f "$BUILDER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker buildx create --name "$BUILDER" --driver docker-container --use >/dev/null
docker buildx inspect --bootstrap >/dev/null
docker buildx build "$ROOT" \
  --builder "$BUILDER" \
  --file "$ROOT/docker/Dockerfile" \
  --platform linux/amd64 \
  --target attention_sink_dev \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg BRANCH_TYPE=local \
  --build-arg BUILD_TYPE=checkpoint-engine \
  --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
  --build-arg BUILD_AND_DOWNLOAD_PARALLEL=4 \
  --build-arg SGLANG_BUILD_COMMIT="$REVISION" \
  --build-arg SGLANG_IMAGE_TAG="$IMAGE" \
  --tag "$IMAGE" \
  --tag "$REVISION_TAG" \
  --push

echo "pushed $IMAGE"
echo "pushed $REVISION_TAG"
