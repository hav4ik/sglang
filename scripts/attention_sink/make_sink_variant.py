#!/usr/bin/env python3
"""Create a copy-on-write checkpoint variant with deterministic sink values."""

import argparse
import json
import os
import signal
import shutil
import struct
import subprocess
import uuid
from pathlib import Path

import torch
from safetensors import safe_open

try:
    from .validate_checkpoint import inspect_checkpoint
except ImportError:
    from validate_checkpoint import inspect_checkpoint


def _read_header(path):
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header_size, header


def _sink_values(name, heads, value, pattern):
    if pattern == "constant":
        return torch.full((heads,), value, dtype=torch.bfloat16)
    if pattern != "ramp":
        raise ValueError(f"unknown sink pattern: {pattern}")
    layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
    return (
        torch.arange(heads, dtype=torch.float32) * 0.25 + value + layer * 0.0625
    ).to(torch.bfloat16)


def _validated_shard_path(model_dir, filename):
    model_dir = model_dir.resolve()
    candidate = model_dir / filename
    if candidate.is_symlink():
        raise ValueError(f"writable checkpoint shard must not be a symlink: {filename}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(model_dir):
        raise ValueError(f"checkpoint shard escapes model directory: {filename}")
    return resolved


def patch_sink_tensors(model_dir, value, pattern="constant"):
    model_dir = Path(model_dir).resolve()
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    config_path = model_dir / "config.json"
    expected_heads = None
    if config_path.exists():
        expected_heads = int(json.loads(config_path.read_text())["num_attention_heads"])
    sink_map = {
        name: filename
        for name, filename in index["weight_map"].items()
        if name.endswith(".self_attn.sinks")
    }
    if not sink_map:
        raise ValueError("checkpoint index contains no attention sink tensors")

    payloads_by_file = {}
    for name, filename in sink_map.items():
        payloads_by_file.setdefault(filename, []).append(name)

    planned_writes = []
    expected_values = {}
    for filename, names in sorted(payloads_by_file.items()):
        path = _validated_shard_path(model_dir, filename)
        header_size, header = _read_header(path)
        for name in sorted(names):
            metadata = header.get(name)
            if metadata is None:
                raise ValueError(f"{name} is missing from {filename}")
            shape = metadata["shape"]
            heads = expected_heads if expected_heads is not None else shape[0]
            if metadata["dtype"] != "BF16" or shape != [heads]:
                raise ValueError(
                    f"unexpected {name}: dtype={metadata['dtype']}, shape={shape}"
                )
            values = _sink_values(name, heads, value, pattern)
            payload = values.view(torch.uint8).numpy().tobytes()
            start, end = metadata["data_offsets"]
            if len(payload) != end - start:
                raise ValueError(f"invalid data offsets for {name}")
            planned_writes.append((path, 8 + header_size + start, payload, name))
            expected_values[name] = values

    patched = 0
    writes_by_file = {}
    for path, offset, payload, name in planned_writes:
        writes_by_file.setdefault(path, []).append((offset, payload, name))
    for path, writes in writes_by_file.items():
        with path.open("r+b", buffering=0) as handle:
            for offset, payload, name in writes:
                handle.seek(offset)
                written = handle.write(payload)
                if written != len(payload):
                    raise OSError(
                        f"short write for {name}: wrote {written} of {len(payload)} bytes"
                    )
                patched += 1
            os.fsync(handle.fileno())

    for filename, names in sorted(payloads_by_file.items()):
        path = _validated_shard_path(model_dir, filename)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in names:
                torch.testing.assert_close(
                    handle.get_tensor(name), expected_values[name], rtol=0, atol=0
                )
    return patched


def clone_reflink(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("source and output must differ")
    if output.is_relative_to(source):
        raise ValueError("output must not be inside the source checkpoint")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    try:
        subprocess.run(
            [
                "cp",
                "--recursive",
                "--dereference",
                "--preserve=mode,timestamps,xattr",
                "--reflink=always",
                f"{source}/.",
                str(output),
            ],
            check=True,
        )
        source_shards = source.glob("*.safetensors")
        for source_shard in source_shards:
            output_shard = output / source_shard.name
            if output_shard.exists() and os.path.samefile(source_shard, output_shard):
                raise RuntimeError(f"reflink clone retained hard link: {source_shard}")
    except Exception:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--value", type=float, default=8.0)
    parser.add_argument("--pattern", choices=("constant", "ramp"), default="ramp")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    source_report = inspect_checkpoint(source)

    def stop_on_signal(signum, _frame):
        raise SystemExit(f"interrupted by signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        clone_reflink(source, temporary)
        shutil.rmtree(temporary / ".cache" / "huggingface", ignore_errors=True)
        count = patch_sink_tensors(temporary, args.value, args.pattern)
        output_report = inspect_checkpoint(temporary)
        source_after = inspect_checkpoint(source)
        if source_after["sink_sha256"] != source_report["sink_sha256"]:
            raise RuntimeError("source checkpoint changed while creating sink variant")
        if output_report["sink_sha256"] == source_report["sink_sha256"]:
            raise RuntimeError("sink variant checksum did not change")
        (temporary / "attention-sink-variant.json").write_text(
            json.dumps(
                {
                    "pattern": args.pattern,
                    "sink_value": args.value,
                    "source": str(source),
                    "source_sink_sha256": source_report["sink_sha256"],
                    "variant_sink_sha256": output_report["sink_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.rename(output)
        output_report["model"] = str(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    print(
        json.dumps(
            {
                "patched_sink_count": count,
                "pattern": args.pattern,
                "sink_value": args.value,
                "source": source_report,
                "variant": output_report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
