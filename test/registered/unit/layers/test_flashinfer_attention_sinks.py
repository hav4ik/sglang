from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.olmo3 import Olmo3Config
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    SGLangBatchAttentionWithAttentionSinkWrapper,
    _run_flashinfer_paged_with_sinks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class _FakeFlashInferWrapper:
    _sglang_sink_window_left = 32

    def run(self, q, paged_kv_cache, *args, **kwargs):
        self.run_args = (q, paged_kv_cache, *args)
        self.run_kwargs = kwargs
        return q


def test_flashinfer_native_sink_run_sets_runtime_options():
    wrapper = _FakeFlashInferWrapper()
    q = torch.randn(2, 4, 8)
    kv_cache = object()
    sinks = torch.randn(4, dtype=torch.float32)

    out = _run_flashinfer_paged_with_sinks(
        wrapper,
        q,
        kv_cache,
        sinks=sinks,
        causal=True,
        sm_scale=0.125,
        window_left=32,
        logits_soft_cap=0.0,
        k_scale=0.5,
        v_scale=0.25,
    )

    assert out is q
    assert wrapper.run_args == (q, kv_cache, sinks, 0.0625)
    assert wrapper.run_kwargs == {
        "v_scale": 0.25,
        "window_left": 32,
    }
    assert wrapper._causal is True


def test_flashinfer_sink_jit_requires_float32_sinks():
    wrapper = _FakeFlashInferWrapper()
    with pytest.raises(TypeError, match="must be float32"):
        _run_flashinfer_paged_with_sinks(
            wrapper,
            torch.randn(2, 4, 8),
            object(),
            sinks=torch.randn(4, dtype=torch.bfloat16),
            window_left=32,
        )


def test_sink_jit_cache_identity_includes_kv_dtype():
    workspace = torch.empty(1, dtype=torch.uint8)
    target = (
        "sglang.srt.layers.attention.flashinfer_backend."
        "BatchPrefillWithPagedKVCacheWrapper.__init__"
    )
    with patch(target, return_value=None) as parent_init:
        SGLangBatchAttentionWithAttentionSinkWrapper(
            workspace,
            backend="fa2",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        bf16_uri = parent_init.call_args.kwargs["jit_args"][0]
        SGLangBatchAttentionWithAttentionSinkWrapper(
            workspace,
            backend="fa2",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )
        fp8_uri = parent_init.call_args.kwargs["jit_args"][0]

    assert bf16_uri != fp8_uri


def test_olmo3_sink_config_detection():
    cases = [
        SimpleNamespace(
            architectures=["Olmo3SinkForCausalLM"],
            model_type="olmo3",
            sink_init_value=None,
        ),
        SimpleNamespace(architectures=["Olmo2ForCausalLM"], model_type="olmo3_sink"),
        SimpleNamespace(
            architectures=["Olmo2ForCausalLM"],
            model_type="olmo3",
            sink_init_value=-10.0,
        ),
    ]

    for hf_config in cases:
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = hf_config
        model_config.hf_text_config = hf_config
        assert model_config._detect_attention_sinks()


def test_yccchen_deploy_config_preserves_sink_architecture():
    hf_config = Olmo3Config(
        architectures=["Olmo3SinkForCausalLM"],
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        sink_init_value=0.0,
        sliding_window=4096,
    )

    assert hf_config.architectures == ["Olmo3SinkForCausalLM"]
    assert hf_config.num_hidden_layers == 64
    assert hf_config.num_attention_heads == 40
    assert hf_config.num_key_value_heads == 8
    assert hf_config.sink_init_value == 0.0


def test_sink_models_disable_ragged_prefill():
    class _Mode:
        @staticmethod
        def is_decode_or_idle():
            return False

        @staticmethod
        def is_target_verify():
            return False

    class _IndicesUpdater:
        def update(self, *args, **kwargs):
            self.kwargs = kwargs

    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.use_sliding_window_kv_pool = False
    backend.is_multimodal = False
    backend.enable_mis = False
    backend.enable_deterministic = False
    backend.use_paged = False
    backend.has_attention_sinks = True
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_split_tile_size = None
    backend.indices_updater_prefill = _IndicesUpdater()

    forward_batch = SimpleNamespace(
        forward_mode=_Mode(),
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        seq_lens_cpu=torch.ones(1, dtype=torch.int32),
        seq_lens_sum=1,
        encoder_lens=None,
        cross_attention_custom_mask=None,
    )

    backend.init_forward_metadata(forward_batch)

    assert backend.indices_updater_prefill.kwargs["use_ragged"] is False
    assert backend.forward_metadata.use_ragged is False


def test_gpt_oss_defaults_to_flashinfer_when_available():
    from sglang.srt.arg_groups import overrides as overrides_module

    server_args = SimpleNamespace(
        is_attention_backend_not_set=lambda: True,
        dtype="auto",
        moe_runner_backend="triton",
    )
    hf_config = SimpleNamespace(quantization_config=None)

    with (
        patch.object(overrides_module, "is_sm100_supported", return_value=False),
        patch.object(overrides_module, "is_sm90_supported", return_value=False),
        patch.object(overrides_module, "is_cpu", return_value=False),
        patch.object(overrides_module, "is_xpu", return_value=False),
        patch.object(overrides_module, "is_hip", return_value=False),
        patch.object(overrides_module, "is_flashinfer_available", return_value=True),
    ):
        result = overrides_module._gpt_oss_overrides(server_args, hf_config)

    assert result["attention_backend"] == "flashinfer"
