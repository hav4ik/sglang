#!/usr/bin/env python3
"""Validate the CUDA 12.8 attention-sink image on an NVIDIA B200 node."""

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_FLASHINFER = "0.6.14"
EXPECTED_CAPABILITY = (10, 0)


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver-major",
        type=int,
        default=570,
        help="required NVIDIA driver major branch (default: 570)",
    )
    args = parser.parse_args()

    cuda_gate = Path(__file__).with_name("check_cuda_128.py")
    if cuda_gate.is_file():
        gate_command = [sys.executable, str(cuda_gate)]
    else:
        installed_gate = shutil.which("sglang-sink-check-cuda")
        if not installed_gate:
            print("CUDA 12.8 gate is unavailable", file=sys.stderr)
            return 1
        gate_command = [installed_gate]
    gate = subprocess.run(gate_command, check=False)
    if gate.returncode:
        return gate.returncode

    errors = []
    report = {
        "required_compute_capability": "10.0",
        "required_driver_major": args.driver_major,
        "required_flashinfer": EXPECTED_FLASHINFER,
    }

    try:
        import flashinfer
        import torch

        flashinfer_version = md.version("flashinfer-python")
        report["flashinfer"] = flashinfer_version
        if flashinfer_version != EXPECTED_FLASHINFER:
            errors.append(
                f"flashinfer-python must be {EXPECTED_FLASHINFER}, "
                f"got {flashinfer_version}"
            )
        if not hasattr(flashinfer, "BatchAttentionWithAttentionSinkWrapper"):
            errors.append("FlashInfer attention-sink wrapper is unavailable")

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["visible_gpu_count"] = torch.cuda.device_count()
        report["gpus"] = []
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            errors.append("no CUDA GPU is visible")
        else:
            for index in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(index)
                gpu = {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                }
                report["gpus"].append(gpu)
                if capability != EXPECTED_CAPABILITY:
                    errors.append(
                        f"GPU {index} must be sm100 for B200 qualification, "
                        f"got sm{capability[0]}{capability[1]} ({gpu['name']})"
                    )

            # Exercise CUDA initialization and a device kernel before reporting success.
            probe = torch.ones(256, device="cuda", dtype=torch.float32)
            report["cuda_probe_sum"] = float(probe.sum().item())
    except Exception as exc:
        errors.append(f"cannot validate B200 Python runtime: {exc}")

    try:
        drivers = command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ).splitlines()
        report["drivers"] = drivers
        for index, driver in enumerate(drivers):
            major = int(driver.strip().split(".", 1)[0])
            if major != args.driver_major:
                errors.append(
                    f"GPU {index} must use R{args.driver_major}, got {driver.strip()}"
                )
    except Exception as exc:
        errors.append(f"cannot validate NVIDIA driver branch: {exc}")

    try:
        checkout = os.environ.get("SGLANG_CHECKOUT")
        if not checkout:
            source_path = Path(__file__).resolve()
            checkout = (
                str(source_path.parents[2])
                if source_path.parent.name == "attention_sink"
                else "/workspace/sglang"
            )
        report["git_revision"] = command(
            "git", "-C", checkout, "rev-parse", "HEAD"
        ).strip()
    except Exception as exc:
        report["git_revision"] = f"unavailable: {exc}"

    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        print("B200 ENVIRONMENT GATE FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("B200 ENVIRONMENT GATE PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
