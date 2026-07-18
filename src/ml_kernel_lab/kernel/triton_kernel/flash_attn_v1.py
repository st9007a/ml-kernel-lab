import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_v1_fwd_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_n: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_n: tl.constexpr,
    n_kv_tiles: tl.constexpr,
    n_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    pad_head_dim: tl.constexpr,
    pad_q_tile_size: tl.constexpr,
    pad_k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    BLOCK_SIZE_M: q tile size
    BLOCK_SIZE_N: k/v tile size
    """
    q_tile_id = tl.program_id(0)  # q tile index
    pid_n = tl.program_id(1)  # batch size, num heads

    b = pid_n // n_heads
    h = pid_n % n_heads

    q_offsets = tl.arange(0, pad_q_tile_size)
    k_offsets = tl.arange(0, pad_k_tile_size)
    d_offsets = tl.arange(0, pad_head_dim)

    q_start = q_tile_id * q_tile_size
    q_end = tl.minimum(q_start + q_tile_size - 1, seq_len - 1)
    q_idx = q_start + q_offsets
    q_tile_mask = (q_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim)

    # load q tile
    q_tile_offsets = b * q_stride_b + h * q_stride_h + q_idx[:, None] * q_stride_n + d_offsets[None, :]
    q_tile = tl.load(q_ptr + q_tile_offsets, mask=q_tile_mask, other=0.)

    # exp sum
    l = tl.zeros((pad_q_tile_size,), dtype=tl.float32)
    # max
    m = tl.full((pad_q_tile_size,), value=-float('inf'), dtype=tl.float32)
    # accumulated output
    #acc = tl.zeros_like(q_tile)
    acc = tl.zeros((pad_q_tile_size, pad_head_dim), dtype=tl.float32)

    if is_causal:
        n_kv_tiles_loop = tl.cdiv(q_end + 1, k_tile_size)
    else:
        n_kv_tiles_loop = n_kv_tiles

    # loop over k/v tiles
    for kv_tile_id in tl.range(0, n_kv_tiles_loop):
        kv_idx = kv_tile_id * k_tile_size + k_offsets

        # load k tile, load in transposed order
        k_tile_offsets = b * k_stride_b + h * k_stride_h + kv_idx[None, :] * k_stride_n + d_offsets[:, None]
        k_tile_mask = (kv_idx[None, :] < seq_len) & (d_offsets[:, None] < head_dim)
        k_tile = tl.load(k_ptr + k_tile_offsets, mask=k_tile_mask, other=0.)

        # load v tile
        v_tile_offsets = b * v_stride_b + h * v_stride_h + kv_idx[:, None] * v_stride_n + d_offsets[None, :]
        v_tile_mask = (kv_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim)
        v_tile = tl.load(v_ptr + v_tile_offsets, mask=v_tile_mask, other=0.)

        # s = q @ k^T
        s_tile = tl.dot(q_tile, k_tile) * sm_scale

        if is_causal:
            s_tile = tl.where((kv_idx[None, :] <= q_idx[:, None]) & (kv_idx[None, :] < seq_len), s_tile, -float('inf'))
        else:
            s_tile = tl.where(kv_idx[None, :] < seq_len, s_tile, -float('inf'))

        m_new = tl.maximum(m, tl.max(s_tile, axis=1))
        p_tile = tl.exp(s_tile - m_new[:, None])
        p_tile = p_tile.to(v_tile.dtype)
        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p_tile, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p_tile, v_tile)

        m = m_new
        l = l_new

    o_tile_offsets = b * o_stride_b + h * o_stride_h + q_idx[:, None] * o_stride_n + d_offsets[None, :]
    o_tile = acc / l[:, None]

    tl.store(out_ptr + o_tile_offsets, o_tile, mask=q_tile_mask)


def flash_attention_v1_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    q = [B, H, Q, D]
    k = [B, H, K, D]
    v = [B, H, K, D]

    # [B, H, Q, K] = [B, H, Q, D] @ [B, H, D, K]
    weights = Q @ K^T / sqrt(head_dim)
    # [B, H, Q, K]
    scores = softmax(weights)
    # [B, H, Q, D] = [B, H, Q, K] @ [B, H, K, D]
    out = scores @ V
    """
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4

    out = torch.empty_like(q)

    B, H, N, D = q.shape

    # TODO: tune tile size and consider SRAM size
    q_tile_size = 128
    k_tile_size = 64
    n_q_tiles = triton.cdiv(N, q_tile_size)
    n_kv_tiles = triton.cdiv(N, k_tile_size)
    grid = (n_q_tiles, B * H)

    flash_attention_v1_fwd_fused_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_kv_tiles,
        H,
        N,
        D,
        1.0 / math.sqrt(D),
        q_tile_size,
        k_tile_size,
        triton.next_power_of_2(D),
        triton.next_power_of_2(q_tile_size),
        triton.next_power_of_2(k_tile_size),
        is_causal,
        num_warps=8,
    )

    return out
