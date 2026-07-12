#!/usr/bin/env python3
"""Validate live sink transfer and disk restoration against a real model."""

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .probe_server import (
        assert_probe_changed,
        assert_probe_parity,
        validate_generation_result,
    )
except ImportError:
    from probe_server import (
        assert_probe_changed,
        assert_probe_parity,
        validate_generation_result,
    )


def probe(engine, lengths, output_tokens):
    results = []
    for length in lengths:
        started = time.time()
        result = engine.generate(
            input_ids=[42] * length,
            sampling_params={"temperature": 0, "max_new_tokens": output_tokens},
            return_logprob=True,
            logprob_start_len=max(length - 1, 0),
            top_logprobs_num=5,
        )
        validate_generation_result(result, length)
        results.append(
            {
                "prompt_length": length,
                "elapsed_s": time.time() - started,
                "output_ids": result.get("output_ids"),
                "meta_info": result.get("meta_info"),
            }
        )
    return results


def require_update_success(result, operation):
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError(f"{operation} returned an invalid result: {result!r}")
    if not result[0]:
        raise RuntimeError(f"{operation} failed: {result[1]}")
    return {"success": bool(result[0]), "message": str(result[1])}


def validate_live_update_quantization(quantization: str) -> None:
    if quantization == "fp8":
        raise ValueError(
            "sink-only tensor updates cannot use the transactional FlashRL loader; "
            "run with --quantization none and qualify FP8 with the complete "
            "A -> B -> A disk-reload cycle"
        )


def sink_checksums(engine, expected_ranks, expected_sinks):
    from sglang.srt.managers.io_struct import CheckWeightsReqInput

    request = CheckWeightsReqInput(action="checksum_attention_sinks")
    result = engine.loop.run_until_complete(
        engine.tokenizer_manager.check_weights(request, None)
    )
    success, message, ranks, per_engine_checksum = result
    if not success or ranks is None or len(ranks) != expected_ranks:
        raise RuntimeError(f"sink checksum failed: {result!r}")
    for rank in ranks:
        if len(rank["checksums"]) != expected_sinks:
            raise RuntimeError(
                f"sink checksum on rank {rank['parallelism_info']} returned "
                f"{len(rank['checksums'])} tensors, expected {expected_sinks}"
            )
    return {
        "message": message,
        "per_engine_checksum": per_engine_checksum,
        "ranks": ranks,
    }


def main():
    import sglang as sgl

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--attention-backend", choices=("triton", "flashinfer"), required=True
    )
    parser.add_argument("--quantization", choices=("none", "fp8"), default="fp8")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--lengths", default="128,4097,16384")
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--sink-value", type=float, default=8.0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--logprob-atol", type=float, default=5e-2)
    args = parser.parse_args()

    validate_live_update_quantization(args.quantization)

    config = json.loads((Path(args.model) / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    if heads % args.tp:
        raise ValueError(f"{heads} attention heads are not divisible by TP={args.tp}")
    sentinel = []
    for layer in range(layers):
        values = (
            torch.arange(heads, dtype=torch.float32) * 0.25
            + args.sink_value
            + layer * 0.0625
        )
        sentinel.append((f"model.layers.{layer}.self_attn.sinks", values))

    engine_args = {
        "model_path": args.model,
        "tp_size": args.tp,
        "attention_backend": args.attention_backend,
        "page_size": 1,
        "load_format": "flash_rl" if args.quantization == "fp8" else "auto",
        "kv_cache_dtype": args.kv_cache_dtype,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "disable_radix_cache": True,
        "skip_tokenizer_init": True,
    }
    if args.quantization != "none":
        engine_args["quantization"] = args.quantization

    lengths = [int(value) for value in args.lengths.split(",") if value]
    report = {"configuration": vars(args)}
    engine = None
    try:
        engine = sgl.Engine(**engine_args)
        report["initial_sink_checksums"] = sink_checksums(engine, args.tp, layers)
        report["initial"] = probe(engine, lengths, args.output_tokens)
        update_result = engine.update_weights_from_tensor(
            sentinel, load_format=None, flush_cache=True
        )
        report["sink_update"] = require_update_success(
            update_result, "live sink update"
        )
        report["updated_sink_checksums"] = sink_checksums(engine, args.tp, layers)
        if (
            report["updated_sink_checksums"]["per_engine_checksum"]
            == report["initial_sink_checksums"]["per_engine_checksum"]
        ):
            raise AssertionError("live sink update did not change sink checksums")
        updated_rank_checksums = {
            rank["per_gpu_checksum"]
            for rank in report["updated_sink_checksums"]["ranks"]
        }
        if len(updated_rank_checksums) != args.tp:
            raise AssertionError("TP ranks received identical sink shards")
        report["after_sink_update"] = probe(engine, lengths, args.output_tokens)
        assert_probe_changed(
            report["initial"], report["after_sink_update"], min_logprob_delta=1e-4
        )

        restore_result = engine.update_weights_from_disk(
            args.model, load_format=engine_args["load_format"]
        )
        report["disk_restore"] = require_update_success(
            restore_result, "checkpoint restore"
        )
        report["restored_sink_checksums"] = sink_checksums(engine, args.tp, layers)
        if (
            report["restored_sink_checksums"]["per_engine_checksum"]
            != report["initial_sink_checksums"]["per_engine_checksum"]
        ):
            raise AssertionError("checkpoint restore did not restore exact sink values")
        report["after_restore"] = probe(engine, lengths, args.output_tokens)
        assert_probe_parity(
            report["initial"], report["after_restore"], args.logprob_atol
        )
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"live sink update validation passed: {args.output}")


if __name__ == "__main__":
    main()
