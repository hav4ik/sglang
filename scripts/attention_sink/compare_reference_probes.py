#!/usr/bin/env python3
"""Compare Triton and FlashInfer server probes to an independent reference."""

import argparse
import json
from pathlib import Path


def load_row(path: Path, prompt_length: int | None = None) -> dict:
    report = json.loads(path.read_text())
    rows = report.get("initial") or []
    if prompt_length is not None:
        rows = [row for row in rows if row.get("prompt_length") == prompt_length]
    if len(rows) != 1:
        suffix = "" if prompt_length is None else f" at prompt length {prompt_length}"
        raise RuntimeError(
            f"expected one initial probe row{suffix} in {path}, got {len(rows)}"
        )
    return rows[0]


def logprobs(row: dict) -> list[float]:
    values = (row.get("meta_info") or {}).get("output_token_logprobs") or []
    if len(values) != len(row.get("output_ids") or []):
        raise RuntimeError("output IDs and logprobs have different lengths")
    return [float(value[0]) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("triton", type=Path)
    parser.add_argument("flashinfer", type=Path)
    args = parser.parse_args()

    reference = load_row(args.reference)
    prompt_length = reference["prompt_length"]
    triton = load_row(args.triton, prompt_length)
    flashinfer = load_row(args.flashinfer, prompt_length)

    reference_lp = logprobs(reference)
    triton_lp = logprobs(triton)
    flashinfer_lp = logprobs(flashinfer)
    if not (len(reference_lp) == len(triton_lp) == len(flashinfer_lp)):
        raise RuntimeError("probe output lengths differ")

    rows = []
    for index, (ref, tri, fi) in enumerate(
        zip(reference_lp, triton_lp, flashinfer_lp, strict=True)
    ):
        rows.append(
            {
                "output_index": index,
                "reference_token": reference["output_ids"][index],
                "triton_token": triton["output_ids"][index],
                "flashinfer_token": flashinfer["output_ids"][index],
                "reference_logprob": ref,
                "triton_logprob": tri,
                "flashinfer_logprob": fi,
                "triton_abs_error": abs(tri - ref),
                "flashinfer_abs_error": abs(fi - ref),
            }
        )

    triton_max = max(row["triton_abs_error"] for row in rows)
    flashinfer_max = max(row["flashinfer_abs_error"] for row in rows)
    closer_backend = (
        "tie"
        if triton_max == flashinfer_max
        else "triton"
        if triton_max < flashinfer_max
        else "flashinfer"
    )
    print(
        json.dumps(
            {
                "prompt_length": prompt_length,
                "reference_output_ids": reference["output_ids"],
                "triton_output_ids": triton["output_ids"],
                "flashinfer_output_ids": flashinfer["output_ids"],
                "triton_max_logprob_abs": triton_max,
                "flashinfer_max_logprob_abs": flashinfer_max,
                "closer_backend_by_max_abs": closer_backend,
                "tokens": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
