"""Long-context hardware qualification for OLMo3 attention sinks."""

import os

import pytest
import torch

from test.registered.attention.test_flashinfer_attention_sink import (
    _eager_reference,
    _flashinfer_attention,
    _triton_decode_attention,
    _triton_extend_attention,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

NUM_Q_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
SWA_WINDOW_LEFT = 4095
MAX_CONTEXT = int(os.environ.get("SINK_LONG_MAX_CONTEXT", "131072"))


def _inputs(seq_len, q_len, kv_dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seq_len + q_len)
    shape = (1, seq_len, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).to(
        kv_dtype
    )
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).to(
        kv_dtype
    )
    q = torch.randn(
        1,
        q_len,
        NUM_Q_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    sinks = torch.linspace(-2, 6, NUM_Q_HEADS, device="cuda", dtype=torch.float32)
    return q, k, v, sinks


def _assert_close(name, actual, expected, *, atol=3e-2, rtol=3e-2):
    max_abs = (actual.float() - expected.float()).abs().max().item()
    print(f"{name}: max_abs={max_abs:.6g}")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_len", [4095, 4096, 4097, 32768, MAX_CONTEXT])
@pytest.mark.parametrize("window_left", [-1, SWA_WINDOW_LEFT])
def test_long_decode_matches_eager(seq_len, window_left):
    q, k, v, sinks = _inputs(seq_len, 1)
    expected = _eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    flashinfer_out = _flashinfer_attention(q, k, v, sinks, window_left)
    triton_out = _triton_decode_attention(q, k, v, sinks, window_left)
    _assert_close("flashinfer/eager decode", flashinfer_out, expected)
    _assert_close("triton/eager decode", triton_out, expected)
    _assert_close("flashinfer/triton decode", flashinfer_out, triton_out)


@pytest.mark.parametrize("seq_len", [4097, 32768, MAX_CONTEXT])
@pytest.mark.parametrize("window_left", [-1, SWA_WINDOW_LEFT])
def test_long_cached_extend_matches_eager(seq_len, window_left):
    q, k, v, sinks = _inputs(seq_len, 4)
    expected = _eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    flashinfer_out = _flashinfer_attention(q, k, v, sinks, window_left)
    triton_out = _triton_extend_attention(q, k, v, sinks, window_left)
    _assert_close("flashinfer/eager extend", flashinfer_out, expected)
    _assert_close("triton/eager extend", triton_out, expected)
    _assert_close("flashinfer/triton extend", flashinfer_out, triton_out)


@pytest.mark.parametrize("window_left", [-1, SWA_WINDOW_LEFT])
def test_sink_extremes_control_denominator(window_left):
    q, k, v, _ = _inputs(4097, 1)
    norms = []
    for sink_value in (-20.0, 0.0, 8.0):
        sinks = torch.full(
            (NUM_Q_HEADS,), sink_value, device="cuda", dtype=torch.float32
        )
        expected = _eager_reference(
            q, k, v, sinks, causal=True, window_left=window_left
        )
        flashinfer_out = _flashinfer_attention(q, k, v, sinks, window_left)
        triton_out = _triton_decode_attention(q, k, v, sinks, window_left)
        _assert_close(f"flashinfer/eager sink={sink_value}", flashinfer_out, expected)
        _assert_close(f"triton/eager sink={sink_value}", triton_out, expected)
        norms.append(expected.float().norm().item())
    assert norms[0] > norms[1] > norms[2], norms


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="E4M3 KV kernels require sm89 or newer",
)
@pytest.mark.parametrize("window_left", [-1, SWA_WINDOW_LEFT])
def test_fp8_kv_sink_attention(window_left):
    q, k, v, sinks = _inputs(4097, 1, torch.float8_e4m3fn)
    expected = _eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    flashinfer_out = _flashinfer_attention(q, k, v, sinks, window_left)
    triton_out = _triton_decode_attention(q, k, v, sinks, window_left)
    _assert_close("flashinfer/eager fp8-kv", flashinfer_out, expected, atol=8e-2)
    _assert_close("triton/eager fp8-kv", triton_out, expected, atol=8e-2)
