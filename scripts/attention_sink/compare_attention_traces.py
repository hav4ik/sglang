#!/usr/bin/env python3
"""Compare captured OLMo attention tensors and replay one layer through each kernel."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.attention_sink.kernel_test_utils import (  # noqa: E402
    eager_reference,
    flashinfer_attention,
    triton_extend_attention,
)


def tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.float()
    right = right.float()
    delta = left - right
    right_norm = torch.linalg.vector_norm(right)
    return {
        "max_abs": torch.max(torch.abs(delta)).item(),
        "rmse": torch.sqrt(torch.mean(delta.square())).item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / right_norm.clamp_min(1e-30)
        ).item(),
    }


def load_trace(root: Path, tp_rank: int, layer_id: int) -> dict:
    path = root / f"tp{tp_rank}" / f"layer-{layer_id:02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def available_layers(root: Path, tp_rank: int) -> set[int]:
    return {
        int(path.stem.removeprefix("layer-"))
        for path in (root / f"tp{tp_rank}").glob("layer-*.pt")
    }


def reshape_trace(trace: dict, device: torch.device) -> tuple:
    token_count = trace["q"].shape[0]
    q = (
        trace["q"]
        .to(device)
        .view(1, token_count, trace["num_q_heads"], trace["head_dim"])
    )
    k = (
        trace["k"]
        .to(device)
        .view(1, token_count, trace["num_kv_heads"], trace["head_dim"])
    )
    v = trace["v"].to(device).view_as(k)
    sinks = trace["sinks"].to(device).float()
    return q, k, v, sinks


def replay(trace: dict, device: torch.device) -> dict:
    q, k, v, sinks = reshape_trace(trace, device)
    window_left = trace["sliding_window_size"]
    expected = eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    triton = triton_extend_attention(q, k, v, sinks, window_left).float()
    flashinfer = flashinfer_attention(q, k, v, sinks, window_left).float()
    production = trace["attention_output"].view_as(q).to(device).float()
    return {
        "triton_vs_eager": tensor_delta(triton, expected),
        "flashinfer_vs_eager": tensor_delta(flashinfer, expected),
        "flashinfer_vs_triton": tensor_delta(flashinfer, triton),
        "production_vs_eager": tensor_delta(production, expected),
        "production_vs_triton": tensor_delta(production, triton),
        "production_vs_flashinfer": tensor_delta(production, flashinfer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("triton_trace", type=Path)
    parser.add_argument("flashinfer_trace", type=Path)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--replay-layer", type=int, default=0)
    args = parser.parse_args()

    triton_layers = available_layers(args.triton_trace, args.tp_rank)
    flashinfer_layers = available_layers(args.flashinfer_trace, args.tp_rank)
    if triton_layers != flashinfer_layers:
        raise RuntimeError(
            f"captured layer sets differ: triton={sorted(triton_layers)}, "
            f"flashinfer={sorted(flashinfer_layers)}"
        )
    if not triton_layers:
        raise RuntimeError("no attention traces found")

    layer_deltas = []
    for layer_id in sorted(triton_layers):
        triton = load_trace(args.triton_trace, args.tp_rank, layer_id)
        flashinfer = load_trace(args.flashinfer_trace, args.tp_rank, layer_id)
        layer_deltas.append(
            {
                "layer_id": layer_id,
                "q": tensor_delta(flashinfer["q"], triton["q"]),
                "k": tensor_delta(flashinfer["k"], triton["k"]),
                "v": tensor_delta(flashinfer["v"], triton["v"]),
                "sinks": tensor_delta(flashinfer["sinks"], triton["sinks"]),
                "attention_output": tensor_delta(
                    flashinfer["attention_output"], triton["attention_output"]
                ),
            }
        )

    if args.replay_layer not in triton_layers:
        raise RuntimeError(f"layer {args.replay_layer} was not captured")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton and FlashInfer replay")
    device = torch.device("cuda")
    triton_replay = replay(
        load_trace(args.triton_trace, args.tp_rank, args.replay_layer), device
    )
    flashinfer_replay = replay(
        load_trace(args.flashinfer_trace, args.tp_rank, args.replay_layer), device
    )

    print(
        json.dumps(
            {
                "tp_rank": args.tp_rank,
                "captured_layers": sorted(triton_layers),
                "cross_backend_layer_deltas": layer_deltas,
                "replay_layer": args.replay_layer,
                "triton_trace_replay": triton_replay,
                "flashinfer_trace_replay": flashinfer_replay,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
