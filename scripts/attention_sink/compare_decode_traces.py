#!/usr/bin/env python3
"""Replay captured OLMo decode attention against Triton, FlashInfer, and eager."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.attention_sink.compare_attention_traces import (  # noqa: E402
    load_trace,
    tensor_delta,
)
from scripts.attention_sink.kernel_test_utils import (  # noqa: E402
    eager_reference,
    flashinfer_attention,
    triton_decode_attention,
)


def load_decode_trace(root: Path, tp_rank: int, layer_id: int, position: int) -> dict:
    path = root / f"tp{tp_rank}" / f"decode-pos-{position}-layer-{layer_id:02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def reshape(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return tensor.view(1, -1, num_heads, head_dim)


def replay_source(
    root: Path,
    tp_rank: int,
    layer_id: int,
    position: int,
    device: torch.device,
    *,
    atol: float,
    rtol: float,
) -> dict:
    prefill = load_trace(root, tp_rank, layer_id)
    prefill_length = prefill["q"].shape[0]
    if position < prefill_length:
        raise ValueError(
            f"decode position {position} precedes prefill length {prefill_length}"
        )

    decode = load_decode_trace(root, tp_rank, layer_id, position)
    k_parts = [prefill["k"]]
    v_parts = [prefill["v"]]
    for decode_position in range(prefill_length, position + 1):
        step = load_decode_trace(root, tp_rank, layer_id, decode_position)
        k_parts.append(step["k"])
        v_parts.append(step["v"])

    num_q_heads = decode["num_q_heads"]
    num_kv_heads = decode["num_kv_heads"]
    head_dim = decode["head_dim"]
    q = reshape(decode["q"], num_q_heads, head_dim).to(device)
    k = reshape(torch.cat(k_parts), num_kv_heads, head_dim).to(device)
    v = reshape(torch.cat(v_parts), num_kv_heads, head_dim).to(device)
    sinks = decode["sinks"].to(device).float()
    window_left = decode["sliding_window_size"]

    eager = eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    triton = triton_decode_attention(q, k, v, sinks, window_left).float()
    flashinfer = flashinfer_attention(q, k, v, sinks, window_left).float()
    production = (
        reshape(decode["attention_output"], num_q_heads, head_dim).to(device).float()
    )

    return {
        "production_vs_triton": tensor_delta(production, triton, atol=atol, rtol=rtol),
        "production_vs_flashinfer": tensor_delta(
            production, flashinfer, atol=atol, rtol=rtol
        ),
        "triton_vs_eager": tensor_delta(triton, eager, atol=atol, rtol=rtol),
        "flashinfer_vs_eager": tensor_delta(flashinfer, eager, atol=atol, rtol=rtol),
        "flashinfer_vs_triton": tensor_delta(flashinfer, triton, atol=atol, rtol=rtol),
    }


def brief(metric: dict) -> dict:
    return {
        "max_abs": metric["max_abs"],
        "relative_l2": metric["relative_l2"],
        "mismatched_elements": metric["mismatched_elements"],
        "total_elements": metric["total_elements"],
        "within_tolerance": metric["within_tolerance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("triton_trace", type=Path)
    parser.add_argument("flashinfer_trace", type=Path)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--positions", default="128,129,130")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton and FlashInfer replay")
    device = torch.device("cuda")
    positions = [int(value) for value in args.positions.split(",") if value]
    rows = []
    checks = {}
    for position in positions:
        triton_source = replay_source(
            args.triton_trace,
            args.tp_rank,
            args.layer,
            position,
            device,
            atol=args.atol,
            rtol=args.rtol,
        )
        flashinfer_source = replay_source(
            args.flashinfer_trace,
            args.tp_rank,
            args.layer,
            position,
            device,
            atol=args.atol,
            rtol=args.rtol,
        )
        position_checks = {
            "triton_native_matches_replay": triton_source["production_vs_triton"][
                "within_tolerance"
            ],
            "flashinfer_native_matches_replay": flashinfer_source[
                "production_vs_flashinfer"
            ]["within_tolerance"],
            "triton_source_triton_matches_eager": triton_source["triton_vs_eager"][
                "within_tolerance"
            ],
            "triton_source_flashinfer_matches_eager": triton_source[
                "flashinfer_vs_eager"
            ]["within_tolerance"],
            "triton_source_backends_match": triton_source["flashinfer_vs_triton"][
                "within_tolerance"
            ],
            "flashinfer_source_triton_matches_eager": flashinfer_source[
                "triton_vs_eager"
            ]["within_tolerance"],
            "flashinfer_source_flashinfer_matches_eager": flashinfer_source[
                "flashinfer_vs_eager"
            ]["within_tolerance"],
            "flashinfer_source_backends_match": flashinfer_source[
                "flashinfer_vs_triton"
            ]["within_tolerance"],
        }
        checks[str(position)] = position_checks
        rows.append(
            {
                "position": position,
                "checks": position_checks,
                "triton_source": {
                    name: brief(metric) for name, metric in triton_source.items()
                },
                "flashinfer_source": {
                    name: brief(metric) for name, metric in flashinfer_source.items()
                },
            }
        )

    passed = all(all(values.values()) for values in checks.values())
    print(
        json.dumps(
            {
                "verdict": {
                    "passed": passed,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "checks": checks,
                },
                "tp_rank": args.tp_rank,
                "layer": args.layer,
                "positions": rows,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit("decode attention trace validation failed")


if __name__ == "__main__":
    main()
