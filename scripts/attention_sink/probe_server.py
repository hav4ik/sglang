#!/usr/bin/env python3
"""Probe a running sink-model server and optionally exercise disk reloads."""

import argparse
import json
import math
import time
from pathlib import Path

import requests


def post(url, endpoint, payload, timeout):
    response = requests.post(f"{url}{endpoint}", json=payload, timeout=timeout)
    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError:
        body = response.text
    if not response.ok:
        raise RuntimeError(f"{endpoint} returned HTTP {response.status_code}: {body!r}")
    return body


def validate_generation_result(result, prompt_length):
    output_ids = result.get("output_ids")
    if not output_ids:
        raise RuntimeError(f"probe at length {prompt_length} returned no output IDs")
    logprobs = (result.get("meta_info") or {}).get("output_token_logprobs") or []
    if not logprobs:
        raise RuntimeError(f"probe at length {prompt_length} returned no logprobs")
    if len(logprobs) != len(output_ids):
        raise RuntimeError(
            f"probe at length {prompt_length} returned {len(output_ids)} IDs but "
            f"{len(logprobs)} logprobs"
        )
    if any(not token or not math.isfinite(float(token[0])) for token in logprobs):
        raise RuntimeError(
            f"probe at length {prompt_length} returned malformed or non-finite logprobs"
        )


def probe(url, model, lengths, output_tokens, timeout):
    results = []
    for length in lengths:
        payload = {
            "input_ids": [42] * length,
            "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens},
            "return_logprob": True,
            "logprob_start_len": max(length - 1, 0),
            "top_logprobs_num": 5,
        }
        started = time.time()
        result = post(url, "/generate", payload, timeout)
        validate_generation_result(result, length)
        row = {
            "model": model,
            "prompt_length": length,
            "elapsed_s": time.time() - started,
            "output_ids": result.get("output_ids"),
            "meta_info": result.get("meta_info"),
        }
        results.append(row)
        print(
            f"probe length={length} output={row['output_ids']} "
            f"elapsed={row['elapsed_s']:.2f}s",
            flush=True,
        )
    return results


def reload(url, model_path, version, timeout, load_format):
    post(url, "/pause_generation", {"mode": "abort"}, timeout)
    result = post(
        url,
        "/update_weights_from_disk",
        {
            "model_path": model_path,
            "weight_version": str(version),
            "flush_cache": True,
            "load_format": load_format,
        },
        timeout,
    )
    if not result.get("success", False):
        raise RuntimeError(f"reload rejected: {result}")
    post(url, "/continue_generation", {}, timeout)
    return result


def get_sink_checksums(url, timeout, expected_count):
    result = post(
        url,
        "/weights_checker",
        {"action": "checksum_attention_sinks"},
        timeout,
    )
    if not result.get("success", False):
        raise RuntimeError(f"sink checksum rejected: {result}")
    ranks = result.get("ranks") or []
    if not ranks:
        raise RuntimeError("sink checksum returned no TP ranks")
    for rank in ranks:
        checksums = rank.get("checksums") or {}
        if len(checksums) != expected_count:
            raise RuntimeError(
                f"sink checksum returned {len(checksums)} tensors on "
                f"rank {rank.get('parallelism_info')}, expected {expected_count}"
            )
    if not result.get("per_engine_checksum"):
        raise RuntimeError("sink checksum returned no engine checksum")
    return result


def assert_probe_parity(expected, actual, logprob_atol):
    if len(expected) != len(actual):
        raise AssertionError("probe result lengths differ")
    for before, after in zip(expected, actual, strict=True):
        length = before["prompt_length"]
        if after["prompt_length"] != length:
            raise AssertionError("probe prompt lengths differ")
        if before["output_ids"] != after["output_ids"]:
            raise AssertionError(f"reload changed greedy output at length {length}")
        before_lp = (before["meta_info"] or {}).get("output_token_logprobs") or []
        after_lp = (after["meta_info"] or {}).get("output_token_logprobs") or []
        if len(before_lp) != len(after_lp):
            raise AssertionError(f"reload changed logprob count at length {length}")
        deltas = []
        for left, right in zip(before_lp, after_lp, strict=True):
            left_value, right_value = float(left[0]), float(right[0])
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise AssertionError(f"non-finite logprob at length {length}")
            deltas.append(abs(left_value - right_value))
        max_abs = max(deltas, default=0.0)
        if max_abs > logprob_atol:
            raise AssertionError(
                f"reload logprob mismatch at length {length}: "
                f"{max_abs} > {logprob_atol}"
            )


def summarize_probe_delta(reference, actual):
    if len(reference) != len(actual):
        raise AssertionError("probe result lengths differ")
    summary = []
    for before, after in zip(reference, actual, strict=True):
        length = before["prompt_length"]
        if after["prompt_length"] != length:
            raise AssertionError("probe prompt lengths differ")
        before_lp = (before["meta_info"] or {}).get("output_token_logprobs") or []
        after_lp = (after["meta_info"] or {}).get("output_token_logprobs") or []
        if len(before_lp) != len(after_lp):
            raise AssertionError(f"probe logprob count differs at length {length}")
        deltas = []
        for left, right in zip(before_lp, after_lp, strict=True):
            left_value, right_value = float(left[0]), float(right[0])
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise AssertionError(f"non-finite logprob at length {length}")
            deltas.append(abs(left_value - right_value))
        summary.append(
            {
                "prompt_length": length,
                "output_ids_equal": before["output_ids"] == after["output_ids"],
                "max_logprob_abs": max(deltas, default=0.0),
            }
        )
    return summary


def assert_probe_changed(before, after, min_logprob_delta):
    if len(before) != len(after):
        raise AssertionError("probe result lengths differ")
    unchanged_lengths = []
    for left, right in zip(before, after, strict=True):
        if left["prompt_length"] != right["prompt_length"]:
            raise AssertionError("probe prompt lengths differ")
        if left["output_ids"] != right["output_ids"]:
            continue
        left_lp = (left["meta_info"] or {}).get("output_token_logprobs") or []
        right_lp = (right["meta_info"] or {}).get("output_token_logprobs") or []
        if len(left_lp) != len(right_lp):
            raise AssertionError(
                "reload changed logprob count without changing output IDs at "
                f"length {left['prompt_length']}"
            )
        deltas = []
        for a, b in zip(left_lp, right_lp, strict=True):
            left_value, right_value = float(a[0]), float(b[0])
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise AssertionError(
                    f"non-finite logprob at length {left['prompt_length']}"
                )
            deltas.append(abs(left_value - right_value))
        if max(deltas, default=0.0) < min_logprob_delta:
            unchanged_lengths.append(left["prompt_length"])
    if unchanged_lengths:
        raise AssertionError(
            "sink-mutated checkpoint did not materially change probes at lengths "
            f"{unchanged_lengths}; required logprob delta >= {min_logprob_delta}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", default="128,4095,4096,4097,16384")
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--reload-model")
    parser.add_argument("--reload-load-format", default="flash_rl")
    parser.add_argument("--warm-reload", action="store_true")
    parser.add_argument("--expected-sink-count", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--logprob-atol", type=float, default=5e-2)
    parser.add_argument("--require-reload-change", action="store_true")
    parser.add_argument("--reload-change-min-logprob", type=float, default=1e-4)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value]
    report = {}
    try:
        cold_initial = probe(
            args.url, args.model, lengths, args.output_tokens, args.timeout
        )
        if args.warm_reload:
            report["cold_initial"] = cold_initial
            report["reload_to_warm_a"] = reload(
                args.url,
                args.model,
                0,
                args.timeout,
                args.reload_load_format,
            )
            report["initial"] = probe(
                args.url, args.model, lengths, args.output_tokens, args.timeout
            )
            report["cold_to_warm_delta"] = summarize_probe_delta(
                report["cold_initial"], report["initial"]
            )
        else:
            report["initial"] = cold_initial
        if args.reload_model:
            report["initial_sink_checksums"] = get_sink_checksums(
                args.url, args.timeout, args.expected_sink_count
            )
            change_error = None
            restore_error = None
            try:
                report["reload_to_b"] = reload(
                    args.url,
                    args.reload_model,
                    1,
                    args.timeout,
                    args.reload_load_format,
                )
                report["after_b"] = probe(
                    args.url,
                    args.reload_model,
                    lengths,
                    args.output_tokens,
                    args.timeout,
                )
                report["after_b_sink_checksums"] = get_sink_checksums(
                    args.url, args.timeout, args.expected_sink_count
                )
                if (
                    report["after_b_sink_checksums"]["per_engine_checksum"]
                    == report["initial_sink_checksums"]["per_engine_checksum"]
                ):
                    raise AssertionError("B reload did not change sink checksums")
                if args.require_reload_change:
                    assert_probe_changed(
                        report["initial"],
                        report["after_b"],
                        args.reload_change_min_logprob,
                    )
            except Exception as exc:
                change_error = exc
                report["change_failure"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            finally:
                try:
                    report["reload_to_a"] = reload(
                        args.url,
                        args.model,
                        2,
                        args.timeout,
                        args.reload_load_format,
                    )
                except Exception as exc:
                    restore_error = exc
                    report["restore_failure"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            if restore_error is not None:
                if change_error is not None:
                    raise RuntimeError(
                        f"B phase failed ({change_error}); A restore also failed "
                        f"({restore_error})"
                    ) from restore_error
                raise restore_error
            report["after_a"] = probe(
                args.url, args.model, lengths, args.output_tokens, args.timeout
            )
            report["after_a_sink_checksums"] = get_sink_checksums(
                args.url, args.timeout, args.expected_sink_count
            )
            if (
                report["after_a_sink_checksums"]["per_engine_checksum"]
                != report["initial_sink_checksums"]["per_engine_checksum"]
            ):
                raise AssertionError("A restore did not restore exact sink checksums")
            assert_probe_parity(report["initial"], report["after_a"], args.logprob_atol)
            if change_error is not None:
                raise change_error
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
