#!/usr/bin/env python3
"""Probe a running sink-model server and optionally exercise disk reloads."""

import argparse
import json
import time
from pathlib import Path

import requests


def post(url, endpoint, payload, timeout):
    response = requests.post(f"{url}{endpoint}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


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


def reload(url, model_path, version, timeout):
    post(url, "/pause_generation", {"mode": "abort"}, timeout)
    result = post(
        url,
        "/update_weights_from_disk",
        {
            "model_path": model_path,
            "weight_version": str(version),
            "flush_cache": True,
        },
        timeout,
    )
    if not result.get("success", False):
        raise RuntimeError(f"reload rejected: {result}")
    post(url, "/continue_generation", {}, timeout)
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
        max_abs = max(
            (
                abs(float(left[0]) - float(right[0]))
                for left, right in zip(before_lp, after_lp, strict=True)
            ),
            default=0.0,
        )
        if max_abs > logprob_atol:
            raise AssertionError(
                f"reload logprob mismatch at length {length}: "
                f"{max_abs} > {logprob_atol}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", default="128,4095,4096,4097,16384")
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--reload-model")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--logprob-atol", type=float, default=5e-2)
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value]
    report = {
        "initial": probe(
            args.url, args.model, lengths, args.output_tokens, args.timeout
        )
    }
    if args.reload_model:
        report["reload_to_b"] = reload(args.url, args.reload_model, 1, args.timeout)
        report["after_b"] = probe(
            args.url, args.reload_model, lengths, args.output_tokens, args.timeout
        )
        report["reload_to_a"] = reload(args.url, args.model, 2, args.timeout)
        report["after_a"] = probe(
            args.url, args.model, lengths, args.output_tokens, args.timeout
        )
        assert_probe_parity(report["initial"], report["after_a"], args.logprob_atol)

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
