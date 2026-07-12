#!/usr/bin/env python3
"""Fail unless every operational CUDA component is CUDA 12.8 or older."""

import ctypes
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import Version


MAX_CUDA = Version("12.8")
NATIVE_BINARY_EXCEPTIONS = {"sglang-kernel": "0.4.4+cu129"}


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def version_from_text(value):
    match = re.search(r"(?<!\d)(\d+\.\d+)", value)
    return Version(match.group(1)) if match else None


def main():
    errors = []
    report = {}

    nvcc = shutil.which("nvcc")
    if not nvcc:
        errors.append("nvcc is missing")
    else:
        nvcc_output = command(nvcc, "--version")
        match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
        nvcc_version = Version(match.group(1)) if match else None
        report["nvcc"] = str(nvcc_version) if nvcc_version else nvcc_output.strip()
        if nvcc_version != MAX_CUDA:
            errors.append(f"nvcc must be exactly 12.8, got {report['nvcc']}")

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        torch_cuda = Version(str(torch.version.cuda))
        if torch_cuda != MAX_CUDA or "+cu128" not in torch.__version__:
            errors.append(
                f"torch must be a +cu128 build reporting CUDA 12.8, got "
                f"{torch.__version__} / {torch.version.cuda}"
            )
    except Exception as exc:
        errors.append(f"cannot validate torch CUDA build: {exc}")

    try:
        runtime = ctypes.CDLL("libcudart.so")
        runtime_version = ctypes.c_int()
        status = runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
        report["cudart"] = runtime_version.value
        if status != 0 or runtime_version.value // 10 != 1208:
            errors.append(
                f"libcudart must report 12080-12089, got status={status}, "
                f"version={runtime_version.value}"
            )
    except Exception as exc:
        errors.append(f"cannot validate libcudart: {exc}")

    # torch 2.11+cu128 requires cuda-bindings>=12.9.4. cuda-python is only a
    # metapackage for those API wrappers; neither package ships the toolkit or
    # runtime libraries. Report them, but do not compare their package version
    # to the operational CUDA ceiling.
    api_binding_packages = {"cuda-python", "cuda-bindings"}
    operational_cuda_packages = {"cuda-core", "cuda-toolkit"}
    package_report = {}
    native_binary_report = {}
    for dist in md.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        version = dist.version
        if not name:
            continue
        newer_tag = re.search(
            r"cu(?:129|13\d)|cuda[-_]?(?:12[._-]?9|13)",
            f"{name}=={version}",
        )
        expected_exception = NATIVE_BINARY_EXCEPTIONS.get(name)
        if expected_exception:
            native_binary_report[name] = version
            if version != expected_exception:
                errors.append(
                    f"{name} must be the audited {expected_exception} wheel, got {version}"
                )
        elif newer_tag:
            errors.append(f"newer CUDA package tag is forbidden: {name}=={version}")
        if name in api_binding_packages | operational_cuda_packages:
            package_report[name] = version
        if name in operational_cuda_packages:
            parsed = version_from_text(version)
            if parsed and parsed > MAX_CUDA:
                errors.append(f"{name} must be <=12.8, got {version}")
    report["cuda_python_api_packages"] = package_report
    report["native_cuda_binary_exceptions"] = native_binary_report
    for name, expected in NATIVE_BINARY_EXCEPTIONS.items():
        if name not in native_binary_report:
            errors.append(
                f"required audited native wheel is missing: {name}=={expected}"
            )

    if shutil.which("dpkg-query"):
        packages = command("dpkg-query", "-W", "-f=${Package}\t${Version}\n")
        bad_dpkg = []
        for line in packages.splitlines():
            name, _, version = line.partition("\t")
            if not re.search(
                r"cuda|cublas|cudnn|cufft|curand|cusolver|cusparse|nccl", name
            ):
                continue
            if re.search(r"(?:cuda)?12[._+-]?9|cuda13|\b13\.", version, re.I):
                bad_dpkg.append(f"{name}={version}")
        report["forbidden_dpkg"] = bad_dpkg
        errors.extend(
            f"newer CUDA Debian package is forbidden: {item}" for item in bad_dpkg
        )

    newer_dirs = [
        str(path)
        for pattern in ("/usr/local/cuda-12.9*", "/usr/local/cuda-13*")
        for path in Path("/").glob(pattern.lstrip("/"))
    ]
    report["forbidden_cuda_directories"] = newer_dirs
    errors.extend(f"newer CUDA directory is forbidden: {path}" for path in newer_dirs)

    # nvidia-smi reports the driver's maximum supported CUDA API, not the
    # container toolkit version, so record it without using it as a gate.
    if shutil.which("nvidia-smi"):
        try:
            report["driver"] = command(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ).strip()
        except subprocess.CalledProcessError as exc:
            report["driver"] = f"unavailable: {exc.output.strip()}"

    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        print("CUDA 12.8 GATE FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("CUDA 12.8 GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
