import json
from unittest.mock import call, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.attention_sink.make_sink_variant import clone_reflink, patch_sink_tensors
from scripts.attention_sink.compare_probes import compare_probe_reports
from scripts.attention_sink.probe_server import (
    assert_probe_changed,
    reload as reload_server,
    summarize_probe_delta,
    validate_generation_result,
)
from scripts.attention_sink.validate_live_sink_update import require_update_success


def _probe(output_ids, logprob):
    return {
        "prompt_length": 128,
        "output_ids": output_ids,
        "meta_info": {"output_token_logprobs": [[logprob, output_ids[0], None]]},
    }


def test_patch_sink_tensors_updates_only_indexed_sinks(tmp_path):
    shard = tmp_path / "model.safetensors"
    sinks = torch.arange(40, dtype=torch.bfloat16)
    untouched = torch.arange(8, dtype=torch.bfloat16)
    save_file(
        {
            "model.layers.0.self_attn.sinks": sinks,
            "model.layers.0.self_attn.q_norm.weight": untouched,
        },
        shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.sinks": shard.name,
                    "model.layers.0.self_attn.q_norm.weight": shard.name,
                }
            }
        )
    )

    assert patch_sink_tensors(tmp_path, 8.0) == 1

    with safe_open(shard, framework="pt", device="cpu") as handle:
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.self_attn.sinks"),
            torch.full((40,), 8.0, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            handle.get_tensor("model.layers.0.self_attn.q_norm.weight"), untouched
        )


def test_patch_sink_tensors_rejects_wrong_shape(tmp_path):
    shard = tmp_path / "model.safetensors"
    name = "model.layers.0.self_attn.sinks"
    save_file({name: torch.zeros(8, dtype=torch.bfloat16)}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name}})
    )
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 40}))

    with pytest.raises(ValueError, match="unexpected"):
        patch_sink_tensors(tmp_path, 8.0)


def test_patch_sink_tensors_rejects_external_shard(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    external = tmp_path / "external.safetensors"
    name = "model.layers.0.self_attn.sinks"
    original = torch.zeros(40, dtype=torch.bfloat16)
    save_file({name: original}, external)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: str(external)}})
    )

    with pytest.raises(ValueError, match="escapes model directory"):
        patch_sink_tensors(model_dir, 8.0)
    with safe_open(external, framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor(name), original)


def test_patch_sink_tensors_ramp_is_layer_and_head_distinct(tmp_path):
    shard = tmp_path / "model.safetensors"
    names = [f"model.layers.{layer}.self_attn.sinks" for layer in (0, 1)]
    save_file({name: torch.zeros(4, dtype=torch.bfloat16) for name in names}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in names}})
    )

    assert patch_sink_tensors(tmp_path, 8.0, "ramp") == 2
    with safe_open(shard, framework="pt", device="cpu") as handle:
        first = handle.get_tensor(names[0])
        second = handle.get_tensor(names[1])
    assert torch.unique(first).numel() == 4
    assert not torch.equal(first, second)


def test_clone_reflink_rejects_output_inside_source(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    with pytest.raises(ValueError, match="must not be inside"):
        clone_reflink(source, source / "variant")


def test_assert_probe_changed_accepts_token_or_logprob_change():
    before = [_probe([1], -1.0)]
    assert_probe_changed(before, [_probe([2], -1.0)], 1e-4)
    assert_probe_changed(before, [_probe([1], -1.01)], 1e-4)


def test_assert_probe_changed_rejects_unchanged_reload():
    before = [_probe([1], -1.0)]
    with pytest.raises(AssertionError, match="did not materially change"):
        assert_probe_changed(before, [_probe([1], -1.0)], 1e-4)

    missing_logprobs = [{**_probe([1], -1.0), "meta_info": {}}]
    with pytest.raises(AssertionError, match="changed logprob count"):
        assert_probe_changed(before, missing_logprobs, 1e-4)


def test_summarize_probe_delta_reports_tokens_and_logprobs():
    summary = summarize_probe_delta(
        [_probe([1], -1.0)],
        [_probe([2], -1.25)],
    )
    assert summary == [
        {
            "prompt_length": 128,
            "output_ids_equal": False,
            "max_logprob_abs": 0.25,
        }
    ]


def test_require_update_success_rejects_failed_or_malformed_results():
    assert require_update_success((True, "ok"), "update") == {
        "success": True,
        "message": "ok",
    }
    with pytest.raises(RuntimeError, match="update failed"):
        require_update_success((False, "bad"), "update")
    with pytest.raises(RuntimeError, match="invalid result"):
        require_update_success(None, "update")


def test_reload_waits_for_idle_flush_before_updating_weights():
    update_result = {"success": True, "message": "ok"}
    with patch(
        "scripts.attention_sink.probe_server.post",
        side_effect=[{"status": "ok"}, "Cache flushed", update_result, {}],
    ) as post_mock:
        result = reload_server("http://server", "/models/b", 7, 1800, "flash_rl")

    assert result is update_result
    assert post_mock.call_args_list == [
        call("http://server", "/pause_generation", {"mode": "abort"}, 1800),
        call("http://server", "/flush_cache?timeout=120", {}, 125.0),
        call(
            "http://server",
            "/update_weights_from_disk",
            {
                "model_path": "/models/b",
                "weight_version": "7",
                "flush_cache": False,
                "load_format": "flash_rl",
            },
            1800,
        ),
        call("http://server", "/continue_generation", {}, 1800),
    ]


def test_validate_generation_result_requires_ids_and_logprobs():
    validate_generation_result(
        {"output_ids": [1], "meta_info": {"output_token_logprobs": [[-1, 1]]}},
        128,
    )
    with pytest.raises(RuntimeError, match="no output IDs"):
        validate_generation_result({"output_ids": []}, 128)
    with pytest.raises(RuntimeError, match="no logprobs"):
        validate_generation_result({"output_ids": [1], "meta_info": {}}, 128)
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_generation_result(
            {
                "output_ids": [1],
                "meta_info": {"output_token_logprobs": [[float("nan"), 1]]},
            },
            128,
        )


def test_compare_probe_reports_checks_every_reload_phase():
    phases = {
        "initial": [_probe([1], -1.0)],
        "after_b": [_probe([2], -2.0)],
        "after_a": [_probe([1], -1.0)],
    }
    comparison = compare_probe_reports(phases, phases, 0.0)
    assert list(comparison) == ["initial", "after_b", "after_a"]

    mismatched = {**phases, "after_b": [_probe([3], -2.0)]}
    with pytest.raises(AssertionError, match="greedy output mismatch"):
        compare_probe_reports(phases, mismatched, 0.0)


def test_compare_probe_reports_rejects_missing_phase_and_nonfinite_logprob():
    phases = {
        "initial": [_probe([1], -1.0)],
        "after_b": [_probe([2], -2.0)],
    }
    with pytest.raises(AssertionError, match="missing from one backend"):
        compare_probe_reports(phases, {"initial": phases["initial"]}, 0.0)

    nonfinite = {"initial": [_probe([1], float("nan"))]}
    with pytest.raises(AssertionError, match="non-finite"):
        compare_probe_reports({"initial": phases["initial"]}, nonfinite, 0.0)
