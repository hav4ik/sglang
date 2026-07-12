"""Shared eager, FlashInfer, and Triton attention-sink test helpers."""

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from sglang.kernels.ops.attention.decode_attention import (  # noqa: E402
    decode_attention_fwd,
)
from sglang.kernels.ops.attention.extend_attention import (  # noqa: E402
    extend_attention_fwd,
)
from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    SGLangBatchAttentionWithAttentionSinkWrapper,
    _run_flashinfer_paged_with_sinks,
)


def eager_reference(q, k, v, sinks, *, causal, window_left=-1):
    repeat = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * (
        q.shape[-1] ** -0.5
    )
    q_pos = torch.arange(q.shape[1], device=q.device) + k.shape[1] - q.shape[1]
    k_pos = torch.arange(k.shape[1], device=q.device)
    mask = torch.zeros(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device)
    if causal:
        mask |= k_pos[None, :] > q_pos[:, None]
    if window_left >= 0:
        mask |= k_pos[None, :] < q_pos[:, None] - window_left
    scores.masked_fill_(mask[None, None], float("-inf"))
    sink_logits = sinks.float().view(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[1], 1)
    probs = torch.cat([scores, sink_logits], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float())


def paged_kv(k, v):
    batch_size, seq_len = k.shape[:2]
    total_tokens = batch_size * seq_len
    kv_indptr = torch.arange(
        0,
        total_tokens + 1,
        seq_len,
        dtype=torch.int32,
        device=k.device,
    )
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=k.device)
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=k.device)
    return (
        (k.flatten(0, 1).unsqueeze(1), v.flatten(0, 1).unsqueeze(1)),
        kv_indptr,
        kv_indices,
        last_page_len,
    )


def flashinfer_attention(q, k, v, sinks, window_left):
    batch_size, _, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    kv_cache, kv_indptr, kv_indices, last_page_len = paged_kv(k, v)
    qo_indptr = torch.arange(
        0,
        batch_size * q.shape[1] + 1,
        q.shape[1],
        dtype=torch.int32,
        device=q.device,
    )
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = SGLangBatchAttentionWithAttentionSinkWrapper(
        workspace,
        "NHD",
        backend="fa2",
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
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
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    return _run_flashinfer_paged_with_sinks(
        wrapper,
        q.flatten(0, 1),
        kv_cache,
        sinks=sinks.float(),
        causal=True,
        sm_scale=head_dim**-0.5,
        window_left=window_left,
    ).view_as(q)


def triton_extend_attention(q, k, v, sinks, window_left):
    batch_size, seq_len, _, head_dim = k.shape
    extend_len = q.shape[1]
    prefix_len = seq_len - extend_len
    q_extend = q.flatten(0, 1).contiguous()
    k_extend = k[:, prefix_len:].to(q.dtype).flatten(0, 1).contiguous()
    v_extend = v[:, prefix_len:].to(q.dtype).flatten(0, 1).contiguous()
    k_buffer = k.flatten(0, 1).contiguous()
    v_buffer = v.flatten(0, 1).contiguous()
    output = torch.empty_like(q_extend)
    qo_indptr = torch.arange(
        0,
        batch_size * extend_len + 1,
        extend_len,
        dtype=torch.int32,
        device=q.device,
    )
    if prefix_len:
        kv_indptr = torch.arange(
            0,
            batch_size * prefix_len + 1,
            prefix_len,
            dtype=torch.int32,
            device=q.device,
        )
        kv_indices = torch.cat(
            [
                torch.arange(
                    batch * seq_len,
                    batch * seq_len + prefix_len,
                    dtype=torch.int32,
                    device=q.device,
                )
                for batch in range(batch_size)
            ]
        )
    else:
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        kv_indices = torch.empty(0, dtype=torch.int32, device=q.device)
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        output,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        extend_len,
        1.0,
        1.0,
        sm_scale=head_dim**-0.5,
        sliding_window_size=window_left,
        sinks=sinks,
    )
    return output.view_as(q)


def triton_decode_attention(q, k, v, sinks, window_left):
    batch_size, seq_len, _, head_dim = k.shape
    num_q_heads = q.shape[2]
    k_buffer = k.flatten(0, 1).contiguous()
    v_buffer = v.flatten(0, 1).contiguous()
    kept = seq_len if window_left < 0 else min(seq_len, window_left + 1)
    kv_indptr = torch.arange(
        0,
        batch_size * kept + 1,
        kept,
        dtype=torch.int32,
        device=q.device,
    )
    kv_indices = torch.cat(
        [
            torch.arange(
                batch * seq_len + seq_len - kept,
                (batch + 1) * seq_len,
                dtype=torch.int32,
                device=q.device,
            )
            for batch in range(batch_size)
        ]
    )
    max_kv_splits = 4
    output = torch.empty_like(q[:, 0])
    attn_logits = torch.empty(
        batch_size,
        num_q_heads,
        max_kv_splits,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        batch_size,
        num_q_heads,
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.full(
        (batch_size,), max_kv_splits, dtype=torch.int32, device=q.device
    )
    decode_attention_fwd(
        q[:, 0].contiguous(),
        k_buffer,
        v_buffer,
        output,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        head_dim**-0.5,
        1.0,
        1.0,
        sinks=sinks,
    )
    return output[:, None]
