#!/usr/bin/env python3
"""Validate all 771 weights in the target OLMo3 sink checkpoint."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def inspect_checkpoint(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if "Olmo3SinkForCausalLM" not in (config.get("architectures") or []):
        raise ValueError("config does not select Olmo3SinkForCausalLM")
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", hidden // heads))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        filename = "model.safetensors"
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            weight_map = {name: filename for name in handle.keys()}

    expected = {
        "model.embed_tokens.weight": None,
        "model.norm.weight": (hidden,),
        "lm_head.weight": None,
    }
    sink_names = set()
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        sink = f"{prefix}.self_attn.sinks"
        sink_names.add(sink)
        expected.update(
            {
                f"{prefix}.self_attn.q_proj.weight": None,
                f"{prefix}.self_attn.k_proj.weight": None,
                f"{prefix}.self_attn.v_proj.weight": None,
                f"{prefix}.self_attn.o_proj.weight": None,
                f"{prefix}.mlp.gate_proj.weight": None,
                f"{prefix}.mlp.up_proj.weight": None,
                f"{prefix}.mlp.down_proj.weight": None,
                sink: (heads,),
                f"{prefix}.self_attn.q_norm.weight": (hidden,),
                f"{prefix}.self_attn.k_norm.weight": (kv_heads * head_dim,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.post_feedforward_layernorm.weight": (hidden,),
            }
        )
    missing = sorted(set(expected) - set(weight_map))
    if missing:
        raise ValueError(f"missing checkpoint tensors: {missing}")

    small_names = {name for name, shape in expected.items() if shape is not None}
    names_by_file = defaultdict(list)
    for name in small_names:
        names_by_file[weight_map[name]].append(name)
    digest = hashlib.sha256()
    sink_values = []
    for filename, names in sorted(names_by_file.items()):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name)
                if (
                    tensor.dtype != torch.bfloat16
                    or tuple(tensor.shape) != expected[name]
                ):
                    raise ValueError(
                        f"invalid {name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
                    )
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite values in {name}")
                if name in sink_names:
                    digest.update(name.encode())
                    digest.update(tensor.view(torch.uint8).numpy().tobytes())
                    sink_values.extend(tensor.float().tolist())
    return {
        "model": str(model_dir.resolve()),
        "checkpoint_weight_count": len(expected),
        "sink_count": len(sink_names),
        "sink_sha256": digest.hexdigest(),
        "sink_min": min(sink_values),
        "sink_max": max(sink_values),
        "sink_mean": sum(sink_values) / len(sink_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = json.dumps(inspect_checkpoint(args.model), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
