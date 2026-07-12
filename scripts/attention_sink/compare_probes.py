#!/usr/bin/env python3
"""Compare deterministic server probes from Triton and FlashInfer."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("triton")
    parser.add_argument("flashinfer")
    parser.add_argument("--logprob-atol", type=float, default=5e-2)
    args = parser.parse_args()
    triton_rows = json.loads(Path(args.triton).read_text())["initial"]
    flashinfer_rows = json.loads(Path(args.flashinfer).read_text())["initial"]
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
        max_abs = max(
            (abs(float(t[0]) - float(f[0])) for t, f in zip(triton_lp, flashinfer_lp)),
            default=0.0,
        )
        if max_abs > args.logprob_atol:
            raise AssertionError(
                f"logprob mismatch at {length}: {max_abs} > {args.logprob_atol}"
            )
        report.append({"prompt_length": length, "max_logprob_abs": max_abs})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
