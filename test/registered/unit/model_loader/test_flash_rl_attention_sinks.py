from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_loader.loader import QuantizedRLModelLoader
from sglang.srt.models.olmo2 import Olmo2ForCausalLM
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class _FakeOlmoModel:
    def __init__(self, params):
        self.params = params

    def named_parameters(self, remove_duplicate=True):
        del remove_duplicate
        return iter(self.params.items())

    def parameters(self):
        return iter(self.params.values())


def test_olmo3_sink_weight_loads_target_tp_shard():
    name = "model.layers.0.self_attn.sinks"
    local_sinks = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))
    model = _FakeOlmoModel({name: local_sinks})
    checkpoint_sinks = torch.arange(40, dtype=torch.bfloat16)

    with patch(
        "sglang.srt.models.olmo2.get_parallel",
        return_value=SimpleNamespace(tp_rank=3, tp_size=8),
    ):
        Olmo2ForCausalLM.load_weights(model, [(name, checkpoint_sinks)])

    torch.testing.assert_close(local_sinks, checkpoint_sinks[15:20].float())
    assert model._last_loaded_attention_sink_names == {name}


def test_olmo3_sink_weight_rejects_wrong_head_count():
    name = "model.layers.0.self_attn.sinks"
    model = _FakeOlmoModel(
        {name: torch.nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))}
    )

    with (
        patch(
            "sglang.srt.models.olmo2.get_parallel",
            return_value=SimpleNamespace(tp_rank=0, tp_size=8),
        ),
        pytest.raises(ValueError, match="expected \\[40\\]"),
    ):
        Olmo2ForCausalLM.load_weights(
            model, [(name, torch.zeros(8, dtype=torch.bfloat16))]
        )


def test_flash_rl_requires_every_sink_on_disk_reload():
    names = [f"model.layers.{i}.self_attn.sinks" for i in range(2)]
    model = _FakeOlmoModel({name: torch.nn.Parameter(torch.zeros(5)) for name in names})

    with pytest.raises(RuntimeError, match="Incomplete OLMo3 checkpoint"):
        QuantizedRLModelLoader._validate_attention_sink_checkpoint(
            model, [(names[0], torch.zeros(10))]
        )

    QuantizedRLModelLoader._validate_attention_sink_checkpoint(
        model, [(name, torch.zeros(10)) for name in names]
    )


def test_flash_rl_requires_olmo_norms_with_sink_reload():
    sink_name = "model.layers.0.self_attn.sinks"
    q_norm_name = "model.layers.0.self_attn.q_norm.weight"
    model = _FakeOlmoModel(
        {
            sink_name: torch.nn.Parameter(torch.zeros(5)),
            q_norm_name: torch.nn.Parameter(torch.zeros(640)),
        }
    )

    with pytest.raises(RuntimeError, match="q_norm.weight"):
        QuantizedRLModelLoader._validate_attention_sink_checkpoint(
            model, [(sink_name, torch.zeros(40))]
        )


def test_initial_load_rejects_missing_sink_tensor():
    names = [f"model.layers.{i}.self_attn.sinks" for i in range(2)]
    model = _FakeOlmoModel({name: torch.nn.Parameter(torch.zeros(5)) for name in names})
    model._last_loaded_attention_sink_names = {names[0]}

    with pytest.raises(RuntimeError, match="were not populated"):
        Olmo2ForCausalLM.validate_loaded_attention_sinks(model)


def test_olmo3_expected_checkpoint_contract_includes_all_projections():
    model = SimpleNamespace(
        config=SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=2,
            attention_bias=False,
        )
    )
    names = Olmo2ForCausalLM.expected_checkpoint_weight_names(model)

    assert len(names) == 27
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names
    assert "model.layers.1.self_attn.q_proj.weight" in names
    assert "model.layers.1.mlp.down_proj.weight" in names


def test_flash_rl_rejects_missing_projection_before_staging():
    sink_name = "model.layers.0.self_attn.sinks"
    model = _FakeOlmoModel({sink_name: torch.nn.Parameter(torch.zeros(5))})
    model.expected_checkpoint_weight_names = lambda: {
        sink_name,
        "model.layers.0.self_attn.q_proj.weight",
    }

    with pytest.raises(RuntimeError, match="q_proj.weight"):
        QuantizedRLModelLoader._validate_attention_sink_checkpoint(
            model, [(sink_name, torch.zeros(40))]
        )


def test_flash_rl_keeps_one_dimensional_sink_in_bfloat16():
    sink = torch.zeros(40, dtype=torch.bfloat16)
    matrix = torch.zeros(40, 128, dtype=torch.bfloat16)

    assert not QuantizedRLModelLoader._should_quantize_weight(sink)
    assert QuantizedRLModelLoader._should_quantize_weight(matrix)


def test_flash_rl_reloads_sink_values_without_reallocating_parameter():
    name = "model.layers.0.self_attn.sinks"
    local_sinks = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))
    model = _FakeOlmoModel({name: local_sinks})
    model.original_weights_rebuild_keys = {
        name: {
            "shape": local_sinks.shape,
            "stride": local_sinks.stride(),
            "dtype": local_sinks.dtype,
            "nbytes": local_sinks.untyped_storage().nbytes(),
        }
    }
    model.recorded_loader = {}
    model.flash_rl_initial_load_complete = True
    original_data_ptr = local_sinks.data_ptr()

    def load_weights(weights):
        Olmo2ForCausalLM.load_weights(model, weights)

    with patch(
        "sglang.srt.models.olmo2.get_parallel",
        return_value=SimpleNamespace(tp_rank=2, tp_size=8),
    ):
        for offset in (0, 100):
            checkpoint_sinks = torch.arange(40, dtype=torch.bfloat16) + offset
            QuantizedRLModelLoader.rebinding_and_load_weights(
                model, load_weights, [(name, checkpoint_sinks)]
            )
            torch.testing.assert_close(local_sinks, checkpoint_sinks[10:15].float())
            assert local_sinks.data_ptr() == original_data_ptr


def test_flash_rl_failed_reload_preserves_sink_value_and_pointer():
    name = "model.layers.0.self_attn.sinks"
    local_sinks = torch.nn.Parameter(torch.full((5,), 7.0, dtype=torch.float32))
    model = _FakeOlmoModel({name: local_sinks})
    model.original_weights_rebuild_keys = {
        name: {
            "shape": local_sinks.shape,
            "stride": local_sinks.stride(),
            "dtype": local_sinks.dtype,
            "nbytes": local_sinks.untyped_storage().nbytes(),
        }
    }
    model.recorded_loader = {}
    model.flash_rl_initial_load_complete = True
    original_value = local_sinks.detach().clone()
    original_data_ptr = local_sinks.data_ptr()

    def failing_load(weights):
        del weights
        local_sinks.data.fill_(99.0)
        raise RuntimeError("injected checkpoint failure")

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        QuantizedRLModelLoader.rebinding_and_load_weights(
            model,
            failing_load,
            [(name, torch.arange(40, dtype=torch.bfloat16))],
        )

    torch.testing.assert_close(local_sinks, original_value)
    assert local_sinks.data_ptr() == original_data_ptr


def test_flash_rl_rejects_missing_stacked_scale_before_commit():
    param_name = "model.layers.0.self_attn.qkv_proj.weight"
    scale_name = "model.layers.0.self_attn.qkv_proj.weight_scale"
    params = {scale_name: torch.nn.Parameter(torch.zeros(1, 6))}
    scales = {"q": torch.ones(2, 1)}

    with (
        patch(
            "sglang.srt.model_loader.loader.get_parallel",
            return_value=SimpleNamespace(tp_rank=0, tp_size=1),
        ),
        pytest.raises(RuntimeError, match="Missing quantized scale shard"),
    ):
        QuantizedRLModelLoader._apply_scale_update(
            params, param_name, scales, validate_only=True
        )

    torch.testing.assert_close(params[scale_name], torch.zeros(1, 6))


def test_flash_rl_applies_gqa_qkv_scales_with_unequal_shard_rows():
    param_name = "model.layers.0.self_attn.qkv_proj.weight"
    scale_name = "model.layers.0.self_attn.qkv_proj.weight_scale"
    params = {scale_name: torch.nn.Parameter(torch.zeros(1, 3584))}
    scales = {
        "q": torch.ones(5120, 1),
        "k": torch.full((1024, 1), 2.0),
        "v": torch.full((1024, 1), 3.0),
    }

    with patch(
        "sglang.srt.model_loader.loader.get_parallel",
        return_value=SimpleNamespace(tp_rank=1, tp_size=2),
    ):
        QuantizedRLModelLoader._apply_scale_update(params, param_name, scales)

    expected = torch.cat(
        [
            torch.ones(1, 2560),
            torch.full((1, 512), 2.0),
            torch.full((1, 512), 3.0),
        ],
        dim=1,
    )
    torch.testing.assert_close(params[scale_name], expected)


def test_flash_rl_does_not_nest_reload_proxy():
    class _Model:
        def __init__(self):
            self.loaded = []

        def load_weights(self, weights):
            self.loaded.extend(weights)

        def named_parameters(self):
            return iter(())

        def named_modules(self):
            return iter(())

    loader = QuantizedRLModelLoader.__new__(QuantizedRLModelLoader)
    model = _Model()
    loader.load_weights_and_postprocess(model, [("first", torch.zeros(1))], "cpu")
    reload_proxy = model.load_weights

    with patch.object(
        QuantizedRLModelLoader, "rebinding_and_load_weights"
    ) as reload_weights:
        loader.load_weights_and_postprocess(model, [("second", torch.ones(1))], "cpu")

    assert model.load_weights is reload_proxy
    reload_weights.assert_called_once()
