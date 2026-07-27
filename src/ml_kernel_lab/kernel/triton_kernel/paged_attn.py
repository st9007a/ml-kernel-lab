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

        # Single-query decode: compute q @ K^T as a vector reduction.
        s = tl.sum(k * q[None, :], axis=1) * sm_scale
        s = tl.where(token_mask, s, -float('inf'))

        m_new = tl.maximum(m, tl.max(s))
        p = tl.exp(s - m_new)
        p = p.to(v.dtype)
        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        m = m_new
        l = l_new

    out = acc / l
    tl.store(out_ptr + pid * o_stride_row + d_offset, out, mask=d_offset < head_dim)


@triton.heuristics(values={
    'BLOCK_SIZE_K': lambda args: triton.next_power_of_2(args['head_dim']),
    'BLOCK_SIZE_N': lambda args: triton.next_power_of_2(args['block_size']),
})
@triton.jit
def single_query_paged_kv_attention_split_kv_fused_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    local_max_ptr,
    local_expsum_ptr,
    out_ptr,
    q_stride_row: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_b: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_b: tl.constexpr,
    bt_stride_row: tl.constexpr,
    local_max_stride_row: tl.constexpr,
    local_expsum_stride_row: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    sm_scale: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_blocks_per_split: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    batch_id = pid_x // num_heads
    hq = pid_x % num_heads
    split_id = pid_y

    seq_len = tl.load(seq_lens_ptr + batch_id)
    num_blocks = tl.cdiv(seq_len, block_size)

    d_offset = tl.arange(0, BLOCK_SIZE_K)
    k_offset = tl.arange(0, BLOCK_SIZE_N)

    q = tl.load(q_ptr + pid_x * q_stride_row + d_offset, mask=d_offset < head_dim, other=0.)

    block_start = split_id * num_blocks_per_split
    block_end = tl.minimum(block_start + num_blocks_per_split, num_blocks)

    local_max = -float('inf')
    local_expsum = 0.
    acc = tl.zeros((BLOCK_SIZE_K, ), dtype=tl.float32)

    for logical_block_idx in tl.range(block_start, block_end):
        physical_block_idx = tl.load(block_table_ptr + batch_id * bt_stride_row + logical_block_idx)

        token_idx = logical_block_idx * block_size + k_offset
        token_mask = token_idx < seq_len

        # load k, v
        k_block_offset = physical_block_idx * k_stride_n + hq * k_stride_h + k_offset[:, None] * k_stride_b + d_offset[None, :]
        k_mask = token_mask[:, None] & (d_offset[None, :] < head_dim)
        k = tl.load(k_cache_ptr + k_block_offset, mask=k_mask, other=0.)

        v_block_offset = physical_block_idx * v_stride_n + hq * v_stride_h + k_offset[:, None] * v_stride_b + d_offset[None, :]
        v = tl.load(v_cache_ptr + v_block_offset, mask=k_mask, other=0.)

        # Single-query decode: compute q @ K^T as a vector reduction.
        s = tl.sum(k * q[None, :], axis=1) * sm_scale
        s = tl.where(token_mask, s, -float('inf'))

        local_max_new = tl.maximum(local_max, tl.max(s))
        p = tl.exp(s - local_max_new).to(v.dtype)
        alpha = tl.exp(local_max - local_max_new)
        local_expsum_new = alpha * local_expsum + tl.sum(p)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        local_max = local_max_new
        local_expsum = local_expsum_new

    tl.store(local_max_ptr + pid_x * local_max_stride_row + split_id, local_max)
    tl.store(local_expsum_ptr + pid_x * local_expsum_stride_row + split_id, local_expsum)
    tl.store(out_ptr + pid_x * out_stride_s + split_id * out_stride_h +  d_offset, acc, mask=d_offset < head_dim)


@triton.heuristics(values={
    'BLOCK_SIZE_M': lambda args: triton.next_power_of_2(args['num_splits']),
    'BLOCK_SIZE_N': lambda args: triton.next_power_of_2(args['head_dim']),
})
@triton.jit
def single_query_paged_kv_attention_reduce_fused_kernel(
    acc_ptr,
    local_max_ptr,
    local_expsum_ptr,
    out_ptr,
    acc_stride_s: tl.constexpr,
    acc_stride_h: tl.constexpr,
    local_max_stride_row: tl.constexpr,
    local_expsum_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    num_splits: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    s_offset = tl.arange(0, BLOCK_SIZE_M)
    d_offset = tl.arange(0, BLOCK_SIZE_N)

    s_mask = s_offset < num_splits
    d_mask = d_offset < head_dim

    acc_offset = pid * acc_stride_s + s_offset[:, None] * acc_stride_h + d_offset[None, :]
    acc = tl.load(acc_ptr + acc_offset, mask=s_mask[:, None] & d_mask[None, :], other=0.)

    local_max = tl.load(local_max_ptr + pid * local_max_stride_row + s_offset, mask=s_mask, other=-float('inf'))
    local_expsum = tl.load(local_expsum_ptr + pid * local_expsum_stride_row + s_offset, mask=s_mask, other=0.)

    global_max = tl.max(local_max)
    alpha = tl.exp(local_max - global_max)
    global_expsum = tl.sum(alpha * local_expsum)

    acc = acc * alpha[:, None]
    out = tl.sum(acc, axis=0) / global_expsum

    tl.store(out_ptr + pid * out_stride_row + d_offset, out, mask=d_mask)


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
        out.stride(0),
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


def single_query_paged_kv_attention_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
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

    num_blocks_per_split = 4
    max_seq_len = int(seq_lens.max().item())
    num_blocks = triton.cdiv(max_seq_len, k_block_size)
    num_splits = triton.cdiv(num_blocks, num_blocks_per_split)

    q_flatten = q.reshape(-1, D)
    acc = torch.empty((B * Hq, num_splits, D), dtype=torch.float32, device=q.device)
    local_max = torch.empty((B * Hq, num_splits), dtype=torch.float32, device=q.device)
    local_expsum = torch.empty((B * Hq, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q_flatten)
    num_warps = 4
    single_query_paged_kv_attention_split_kv_fused_kernel[(B * Hq, num_splits)](
        q_flatten,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        local_max,
        local_expsum,
        acc,
        q_flatten.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        local_max.stride(0),
        local_expsum.stride(0),
        acc.stride(0),
        acc.stride(1),
        1 / math.sqrt(D),
        Hq,
        D,
        k_block_size,
        num_blocks_per_split,
        num_warps=num_warps,
    )

    single_query_paged_kv_attention_reduce_fused_kernel[(B * Hq, )](
        acc,
        local_max,
        local_expsum,
        out,
        acc.stride(0),
        acc.stride(1),
        local_max.stride(0),
        local_expsum.stride(0),
        out.stride(0),
        num_splits,
        D,
        num_warps=num_warps
    )

    return out.reshape(q.shape)
