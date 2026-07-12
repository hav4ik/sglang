#!/usr/bin/env python3
"""Compare deterministic server probes from Triton and FlashInfer."""

import argparse
import json
import math
from pathlib import Path


def compare_probe_rows(triton_rows, flashinfer_rows, logprob_atol):
    if len(triton_rows) != len(flashinfer_rows):
        raise AssertionError("probe result lengths differ")

    report = []
    for triton, flashinfer in zip(triton_rows, flashinfer_rows, strict=True):
        length = triton["prompt_length"]
        if flashinfer["prompt_length"] != length:
            raise AssertionError("prompt lengths differ")
        if triton["output_ids"] != flashinfer["output_ids"]:
            raise AssertionError(
                f"greedy output mismatch at {length}: "
                f"triton={triton['output_ids']}, flashinfer={flashinfer['output_ids']}"
            )
        triton_lp = (triton["meta_info"] or {}).get("output_token_logprobs") or []
        flashinfer_lp = (flashinfer["meta_info"] or {}).get(
            "output_token_logprobs"
        ) or []
        if len(triton_lp) != len(flashinfer_lp):
            raise AssertionError(f"logprob lengths differ at {length}")
        deltas = []
        for triton_token, flashinfer_token in zip(
            triton_lp, flashinfer_lp, strict=True
        ):
            triton_value = float(triton_token[0])
            flashinfer_value = float(flashinfer_token[0])
            if not math.isfinite(triton_value) or not math.isfinite(flashinfer_value):
                raise AssertionError(f"non-finite logprob at {length}")
            deltas.append(abs(triton_value - flashinfer_value))
        max_abs = max(deltas, default=0.0)
        if max_abs > logprob_atol:
            raise AssertionError(
                f"logprob mismatch at {length}: {max_abs} > {logprob_atol}"
            )
        report.append({"prompt_length": length, "max_logprob_abs": max_abs})
    return report


def compare_probe_reports(triton_report, flashinfer_report, logprob_atol):
    phases = ("initial", "after_b", "after_a")
    comparison = {}
    for phase in phases:
        triton_present = phase in triton_report
        flashinfer_present = phase in flashinfer_report
        if triton_present != flashinfer_present:
            raise AssertionError(f"probe phase {phase!r} is missing from one backend")
        if triton_present:
            comparison[phase] = compare_probe_rows(
                triton_report[phase], flashinfer_report[phase], logprob_atol
            )
    if "initial" not in comparison:
        raise AssertionError("probe reports are missing the initial phase")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("triton")
    parser.add_argument("flashinfer")
    parser.add_argument("--logprob-atol", type=float, default=5e-2)
    args = parser.parse_args()
    triton_report = json.loads(Path(args.triton).read_text())
    flashinfer_report = json.loads(Path(args.flashinfer).read_text())
    comparison = compare_probe_reports(
        triton_report, flashinfer_report, args.logprob_atol
    )
    comparison["_metadata"] = {"logprob_atol": args.logprob_atol}
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
