import math

import torch
import triton
import triton.language as tl


@triton.jit
def single_query_paged_kv_attention_fused_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    q_stride_row: tl.constexpr,
    o_stride_row: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_b: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_b: tl.constexpr,
    bt_stride_row: tl.constexpr,
    sm_scale: tl.constexpr,
    num_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    pad_block_size: tl.constexpr,
    pad_head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    hq = pid % num_heads

    seq_len = tl.load(seq_lens_ptr + batch_id)
    num_blocks = tl.cdiv(seq_len, block_size)

    d_offset = tl.arange(0, pad_head_dim)
    k_offset = tl.arange(0, pad_block_size)
    q = tl.load(q_ptr + pid * q_stride_row + d_offset, mask=d_offset < head_dim, other=0.)
    q = q.view(1, pad_head_dim)

    # exp sum
    l = 0.
    # max
    m = -float('inf')
    # accumulated output
    acc = tl.zeros((pad_head_dim, ), dtype=tl.float32)

    for logical_block_idx in tl.range(0, num_blocks):
        physical_block_idx = tl.load(block_table_ptr + batch_id * bt_stride_row + logical_block_idx)

        token_idx = logical_block_idx * block_size + k_offset
        token_mask = token_idx < seq_len

        # load k, v
        k_block_offset = physical_block_idx * k_stride_n + hq * k_stride_h + k_offset[:, None] * k_stride_b + d_offset[None, :]
        k_mask = token_mask[:, None] & (d_offset[None, :] < head_dim)
        k = tl.load(k_cache_ptr + k_block_offset, mask=k_mask, other=0.)

        v_block_offset = physical_block_idx * v_stride_n + hq * v_stride_h + k_offset[:, None] * v_stride_b + d_offset[None, :]
        v = tl.load(v_cache_ptr + v_block_offset, mask=k_mask, other=0.)

        # flash attn v1 style implementation
        s = tl.dot(q, tl.trans(k, (1, 0))) * sm_scale
        s = tl.where(token_mask[None, :], s, -float('inf'))

        m_new = tl.maximum(m, tl.max(s))
        p = tl.exp(s - m_new)
        p = p.to(v.dtype)
        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p)

        acc = acc * alpha + tl.dot(p, v).view(pad_head_dim)

        m = m_new
        l = l_new

    out = acc / l
    tl.store(out_ptr + pid * o_stride_row + d_offset, out, mask=d_offset < head_dim)


def single_query_paged_kv_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """
    q = [B, Hq, D]
    k/v_cache = [num_blocks, Hkv, block_size, D]
    block_table = [B, max_blocks_per_seq]
    seq_lens = [B]
    out = [B, Hq, D]
    """
    assert q.dim() == 3
    assert k_cache.dim() == v_cache.dim() == 4
    assert block_table.dim() == 2
    assert seq_lens.dim() == 1

    B, Hq, D = q.shape
    _, Hk, k_block_size, Dk = k_cache.shape
    _, Hv, v_block_size, Dv = v_cache.shape
    Bbt, _ = block_table.shape
    Bs = seq_lens.shape[0]

    assert k_block_size == v_block_size
    assert B == Bbt == Bs
    assert Hk == Hv == Hq
    assert D == Dk == Dv

    assert torch.all(seq_lens > 0).item()

    q_flatten = q.reshape(-1, D)
    out = torch.empty_like(q_flatten)
    num_warps = 4
    grid = (B * Hq, )
    single_query_paged_kv_attention_fused_kernel[grid](
        q_flatten,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        q_flatten.stride(0),
        out.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        1 / math.sqrt(D),
        Hq,
        k_block_size,
        D,
        triton.next_power_of_2(k_block_size),
        triton.next_power_of_2(D),
        num_warps=num_warps,
    )

    return out.reshape(q.shape)
