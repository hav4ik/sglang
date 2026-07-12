#!/usr/bin/env python3
"""Probe a checkpoint with Yi-Chia Chen's exact eager attention-sink model."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

YICHIA_REVISION = "bc03a2c71a076990deaad3d712c6889682e12c69"


def require_source(source: Path) -> None:
    required = source / "olmo3_sink" / "modeling_olmo3_sink.py"
    if not required.is_file():
        raise FileNotFoundError(f"Yi-Chia source is missing {required}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != YICHIA_REVISION:
        raise RuntimeError(
            f"expected Yi-Chia revision {YICHIA_REVISION}, got {revision}"
        )


def sink_summary(model, model_dir: str) -> dict:
    parameters = dict(model.named_parameters())
    sink_names = {name for name in parameters if name.endswith(".self_attn.sinks")}
    if len(sink_names) != 64:
        raise RuntimeError(f"expected 64 attention-sink tensors, got {len(sink_names)}")

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names_by_file = defaultdict(list)
    for name in sink_names:
        names_by_file[index[name]].append(name)

    digest = hashlib.sha256()
    values = []
    for filename, names in sorted(names_by_file.items()):
        for name in sorted(names):
            parameter = parameters[name]
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"expected BF16 runtime sink {name}, got {parameter.dtype}"
                )
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            values.extend(tensor.float().tolist())

    return {
        "count": len(sink_names),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "checkpoint_compatible_sha256": digest.hexdigest(),
    }


@torch.inference_mode()
def probe(model, prompt_length: int, output_tokens: int) -> dict:
    embedding_device = model.get_input_embeddings().weight.device
    input_ids = torch.full(
        (1, prompt_length), 42, dtype=torch.long, device=embedding_device
    )
    output_ids = []
    output_token_logprobs = []
    past_key_values = None
    current_ids = input_ids
    for _ in range(output_tokens):
        output = model(
            input_ids=current_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[:, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        next_token = torch.argmax(logits, dim=-1)
        token = int(next_token.item())
        output_ids.append(token)
        output_token_logprobs.append([float(logprobs[0, token].item()), token])
        past_key_values = output.past_key_values
        current_ids = next_token.to(embedding_device).view(1, 1)

    return {
        "prompt_length": prompt_length,
        "output_ids": output_ids,
        "meta_info": {"output_token_logprobs": output_token_logprobs},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--max-memory", default="70GiB")
    args = parser.parse_args()

    require_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))

    import accelerate  # noqa: F401, PLC0415
    import transformers  # noqa: PLC0415
    from olmo3_sink.configuration_olmo3_sink import (  # noqa: PLC0415
        Olmo3SinkConfig,
    )
    from olmo3_sink.modeling_olmo3_sink import (  # noqa: PLC0415
        Olmo3SinkForCausalLM,
    )

    if not transformers.__version__.startswith("5.9."):
        raise RuntimeError(
            "Yi-Chia eager reference requires transformers 5.9.x, got "
            f"{transformers.__version__}"
        )
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected exactly two visible GPUs, got {torch.cuda.device_count()}"
        )

    config = Olmo3SinkConfig.from_pretrained(args.model, local_files_only=True)
    config._attn_implementation = "eager"
    max_memory = {device: args.max_memory for device in range(2)}
    model = Olmo3SinkForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    report = {
        "reference": "yichia-eager",
        "source_revision": YICHIA_REVISION,
        "model": args.model,
        "transformers": transformers.__version__,
        "sink_summary": sink_summary(model, args.model),
        "initial": [probe(model, args.length, args.output_tokens)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
