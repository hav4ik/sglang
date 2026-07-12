import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from scripts.attention_sink.kernel_test_utils import (  # noqa: E402
    eager_reference,
    flashinfer_attention,
    paged_kv,
    triton_decode_attention,
    triton_extend_attention,
)
from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    SGLangBatchAttentionWithAttentionSinkWrapper,
    _run_flashinfer_paged_with_sinks,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b", runner_config="4-gpu-b200")


@pytest.mark.parametrize(
    "mode,extend_len", [("prefill", 16), ("extend", 5), ("decode", 1)]
)
@pytest.mark.parametrize("window_left", [-1, 4])
@pytest.mark.parametrize(
    "sink_value",
    [None, 13.6875, 21.75],
    ids=["random", "checkpoint-max", "variant-max"],
)
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16-kv", "fp8-kv"]
)
def test_flashinfer_and_triton_attention_sinks_match_eager(
    mode, extend_len, window_left, sink_value, kv_dtype
):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    if kv_dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("E4M3 KV kernels require sm89 or newer")

    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 16, 40, 8, 128
    k = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype
    ).to(kv_dtype)
    v = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype
    ).to(kv_dtype)
    q = torch.randn(
        batch_size, extend_len, num_q_heads, head_dim, device=device, dtype=dtype
    )
    sinks = (
        torch.randn(num_q_heads, device=device, dtype=dtype)
        if sink_value is None
        else torch.full((num_q_heads,), sink_value, device=device, dtype=torch.float32)
    )

    flashinfer_out = flashinfer_attention(q, k, v, sinks, window_left)
    expected = eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    tolerance = 8e-2 if kv_dtype == torch.float8_e4m3fn else 3e-2
    torch.testing.assert_close(
        flashinfer_out.float(), expected, rtol=tolerance, atol=tolerance
    )

    if mode == "decode":
        triton_out = triton_decode_attention(q, k, v, sinks, window_left)
    else:
        triton_out = triton_extend_attention(q, k, v, sinks, window_left)

    torch.testing.assert_close(
        triton_out.float(), expected, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        flashinfer_out.float(),
        triton_out.float(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("window_left", [-1, 4])
def test_fp8_kv_attention_sinks_apply_nonunit_scales(window_left):
    torch.manual_seed(2)
    device = "cuda"
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("E4M3 KV kernels require sm89 or newer")

    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 16, 40, 8, 128
    q = torch.randn(
        batch_size, 1, num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_scale, v_scale = 0.25, 1.75
    k_cache = (
        torch.randn(
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        ).float()
        / k_scale
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        ).float()
        / v_scale
    ).to(torch.float8_e4m3fn)
    sinks = torch.linspace(
        -0.30859375, 13.6875, num_q_heads, device=device, dtype=torch.float32
    )
    expected = eager_reference(
        q,
        k_cache.float() * k_scale,
        v_cache.float() * v_scale,
        sinks,
        causal=True,
        window_left=window_left,
    )
    flashinfer_out = flashinfer_attention(
        q,
        k_cache,
        v_cache,
        sinks,
        window_left,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    triton_out = triton_decode_attention(
        q,
        k_cache,
        v_cache,
        sinks,
        window_left,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(flashinfer_out.float(), expected, rtol=0.08, atol=0.08)
    torch.testing.assert_close(triton_out.float(), expected, rtol=0.08, atol=0.08)
    torch.testing.assert_close(
        flashinfer_out.float(), triton_out.float(), rtol=0.08, atol=0.08
    )


@pytest.mark.parametrize("window_left", [-1, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize(
    "kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16-kv", "fp8-kv"]
)
def test_flashinfer_attention_sinks_cuda_graph_reads_reloaded_values(
    window_left, batch_size, kv_dtype
):
    torch.manual_seed(1)
    device, dtype = "cuda", torch.bfloat16
    if kv_dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("E4M3 KV kernels require sm89 or newer")
    seq_len, num_q_heads, num_kv_heads, head_dim = 16, 40, 8, 128
    q = torch.randn(batch_size, 1, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype
    ).to(kv_dtype)
    v = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype
    ).to(kv_dtype)
    sinks = torch.zeros(num_q_heads, device=device, dtype=torch.float32)
    kv_cache, kv_indptr, kv_indices, last_page_len = paged_kv(k, v)
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = SGLangBatchAttentionWithAttentionSinkWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr,
        paged_kv_indptr_buf=kv_indptr,
        paged_kv_indices_buf=kv_indices,
        paged_kv_last_page_len_buf=last_page_len,
        backend="fa2",
        q_data_type=dtype,
        kv_data_type=kv_dtype,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        window_left=window_left,
    )
    wrapper._sglang_sink_window_left = window_left
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=True,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=kv_dtype,
    )

    def run():
        return _run_flashinfer_paged_with_sinks(
            wrapper,
            q.flatten(0, 1),
            kv_cache,
            sinks=sinks,
            causal=True,
            sm_scale=head_dim**-0.5,
            window_left=window_left,
        ).view_as(q)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    sinks.fill_(6.0)
    graph.replay()
    expected = eager_reference(q, k, v, sinks, causal=True, window_left=window_left)
    tolerance = 8e-2 if kv_dtype == torch.float8_e4m3fn else 3e-2
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)
