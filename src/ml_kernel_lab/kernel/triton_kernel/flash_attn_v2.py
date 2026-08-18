import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_inner_loop(
    q_tile,
    k_base_ptr,
    v_base_ptr,
    out_acc,
    row_sum,
    row_max,
    offs_n,
    offs_d,
    q_rows,
    range_start_n,
    range_end_n,
    k_stride_row,
    v_stride_row,
    seq_len,
    head_dim: tl.constexpr,
    sm_scale,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL_MASK: tl.constexpr,
    CHECK_KV_BOUNDS: tl.constexpr,
    CHECK_D_BOUNDS: tl.constexpr,
):
    d_mask = offs_d[None, :] < head_dim

    for start_n in tl.range(range_start_n, range_end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # load k tile
        kv_rows = start_n + offs_n
        kv_mask = kv_rows[:, None] < seq_len
        k_ptrs = k_base_ptr + kv_rows[:, None] * k_stride_row + offs_d[None, :]

        if CHECK_KV_BOUNDS and CHECK_D_BOUNDS:
            k_tile = tl.load(k_ptrs, mask=kv_mask & d_mask, other=0.)
        elif CHECK_D_BOUNDS:
            k_tile = tl.load(k_ptrs, mask=d_mask, other=0.)
        elif CHECK_KV_BOUNDS:
            k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.)
        else:
            k_tile = tl.load(k_ptrs)

        # s = q @ k^T
        scores = tl.dot(q_tile, tl.trans(k_tile, (1, 0))) * sm_scale

        if APPLY_CAUSAL_MASK:
            score_mask = kv_rows[None, :] <= q_rows[:, None]
            if CHECK_KV_BOUNDS:
                score_mask &= kv_rows[None, :] < seq_len
            scores = tl.where(score_mask, scores, -float('inf'))
        elif CHECK_KV_BOUNDS:
            score_mask = kv_rows[None, :] < seq_len
            scores = tl.where(score_mask, scores, -float('inf'))

        row_max_new = tl.maximum(row_max, tl.max(scores, axis=1))
        probs = tl.exp(scores - row_max_new[:, None])

        # load v tile
        v_ptrs = v_base_ptr + kv_rows[:, None] * v_stride_row + offs_d[None, :]

        if CHECK_KV_BOUNDS and CHECK_D_BOUNDS:
            v_tile = tl.load(v_ptrs, mask=kv_mask & d_mask, other=0.)
        elif CHECK_D_BOUNDS:
            v_tile = tl.load(v_ptrs, mask=d_mask, other=0.)
        elif CHECK_KV_BOUNDS:
            v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.)
        else:
            v_tile = tl.load(v_ptrs)

        rescale = tl.exp(row_max - row_max_new)
        row_sum_new = rescale * row_sum + tl.sum(probs, axis=1)

        probs = probs.to(v_tile.dtype)
        out_acc = out_acc * rescale[:, None] + tl.dot(probs, v_tile)

        row_max = row_max_new
        row_sum = row_sum_new

    return out_acc, row_sum, row_max


@triton.heuristics(
    {
        'EVEN_M': lambda args: args['seq_len'] % args['BLOCK_M'] == 0,
        'EVEN_N': lambda args: args['seq_len'] % args['BLOCK_N'] == 0,
        'EVEN_D': lambda args: args['head_dim'] == args['BLOCK_D'],
    }
)
@triton.jit
def flash_attention_v2_fwd_fused_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_n,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    o_stride_b,
    o_stride_h,
    o_stride_n,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    seq_len,
    head_dim: tl.constexpr,
    sm_scale,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)  # q tile index
    pid_bh = tl.program_id(1)  # batch size, num heads

    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    kv_head_idx = q_head_idx // (n_q_heads // n_kv_heads)

    q_base_ptr = q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h
    k_base_ptr = k_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h
    v_base_ptr = v_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    start_m = pid_m * BLOCK_M
    start_m = tl.multiple_of(start_m, BLOCK_M)

    q_rows = start_m + offs_m
    q_ptrs = q_base_ptr + q_rows[:, None] * q_stride_n + offs_d[None, :]
    q_mask = q_rows[:, None] < seq_len

    d_mask = offs_d[None, :] < head_dim

    # load q tile
    if EVEN_M and EVEN_D:
        q_tile = tl.load(q_ptrs)
    elif EVEN_M:
        q_tile = tl.load(q_ptrs, mask=d_mask, other=0.)
    elif EVEN_D:
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.)
    else:
        q_tile = tl.load(q_ptrs, mask=q_mask & d_mask, other=0.)

    # exp sum
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # max
    row_max = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    # accumulated output
    out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if is_causal:
        out_acc, row_sum, row_max = flash_attention_inner_loop(
            q_tile,
            k_base_ptr,
            v_base_ptr,
            out_acc,
            row_sum,
            row_max,
            offs_n,
            offs_d,
            q_rows,
            0,
            start_m,
            k_stride_n,
            v_stride_n,
            seq_len,
            head_dim,
            sm_scale,
            BLOCK_N=BLOCK_N,
            APPLY_CAUSAL_MASK=False,
            CHECK_KV_BOUNDS=False,
            CHECK_D_BOUNDS=not EVEN_D,
        )

        diagonal_end_n = tl.minimum(start_m + BLOCK_M, seq_len)

        out_acc, row_sum, row_max = flash_attention_inner_loop(
            q_tile,
            k_base_ptr,
            v_base_ptr,
            out_acc,
            row_sum,
            row_max,
            offs_n,
            offs_d,
            q_rows,
            start_m,
            diagonal_end_n,
            k_stride_n,
            v_stride_n,
            seq_len,
            head_dim,
            sm_scale,
            BLOCK_N=BLOCK_N,
            APPLY_CAUSAL_MASK=True,
            CHECK_KV_BOUNDS=not EVEN_N,
            CHECK_D_BOUNDS=not EVEN_D,
        )
    else:
        out_acc, row_sum, row_max = flash_attention_inner_loop(
            q_tile,
            k_base_ptr,
            v_base_ptr,
            out_acc,
            row_sum,
            row_max,
            offs_n,
            offs_d,
            q_rows,
            0,
            seq_len,
            k_stride_n,
            v_stride_n,
            seq_len,
            head_dim,
            sm_scale,
            BLOCK_N=BLOCK_N,
            APPLY_CAUSAL_MASK=False,
            CHECK_KV_BOUNDS=not EVEN_N,
            CHECK_D_BOUNDS=not EVEN_D,
        )

    out_tile = out_acc / row_sum[:, None]

    out_base_ptr = out_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h
    out_ptrs = out_base_ptr + q_rows[:, None] * o_stride_n + offs_d[None, :]

    if EVEN_M and EVEN_D:
        tl.store(out_ptrs, out_tile)
    elif EVEN_M:
        tl.store(out_ptrs, out_tile, mask=d_mask)
    elif EVEN_D:
        tl.store(out_ptrs, out_tile, mask=q_mask)
    else:
        tl.store(out_ptrs, out_tile, mask=q_mask & d_mask)


@triton.heuristics(
    {
        'EVEN_M': lambda args: args['seq_len'] % args['q_tile_size'] == 0,
        'EVEN_N': lambda args: args['seq_len'] % args['k_tile_size'] == 0,
        'EVEN_HEADDIM': lambda args: args['head_dim'] == args['pad_head_dim'],
    }
)
@triton.jit
def flash_attention_v2_gqa_fwd_fused_kernel(
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
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    q_tile_size: tl.constexpr,
    k_tile_size: tl.constexpr,
    pad_head_dim: tl.constexpr,
    pad_q_tile_size: tl.constexpr,
    pad_k_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    """
    BLOCK_SIZE_M: q tile size
    BLOCK_SIZE_N: k/v tile size
    """
    q_tile_id = tl.program_id(0)  # q tile index
    pid_n = tl.program_id(1)  # batch size, num heads

    b = pid_n // n_q_heads
    hq = pid_n % n_q_heads
    hk = hq // (n_q_heads // n_kv_heads)

    q_offsets = tl.arange(0, pad_q_tile_size)
    k_offsets = tl.arange(0, pad_k_tile_size)
    d_offsets = tl.arange(0, pad_head_dim)

    q_start = q_tile_id * q_tile_size
    q_start = tl.multiple_of(q_start, q_tile_size)
    q_end = tl.minimum(q_start + q_tile_size - 1, seq_len - 1)
    q_idx = q_start + q_offsets
    q_tile_offsets = b * q_stride_b + hq * q_stride_h + q_idx[:, None] * q_stride_n + d_offsets[None, :]

    # load q tile
    if EVEN_M and EVEN_HEADDIM:
        q_tile = tl.load(q_ptr + q_tile_offsets)
    elif EVEN_M:
        q_tile = tl.load(q_ptr + q_tile_offsets, mask=d_offsets[None, :] < head_dim, other=0.)
    elif EVEN_HEADDIM:
        q_tile = tl.load(q_ptr + q_tile_offsets, mask=q_idx[:, None] < seq_len, other=0.)
    else:
        q_tile = tl.load(q_ptr + q_tile_offsets, mask=(q_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim), other=0.)

    # exp sum
    l = tl.zeros((pad_q_tile_size,), dtype=tl.float32)
    # max
    m = tl.full((pad_q_tile_size,), value=-float('inf'), dtype=tl.float32)
    # accumulated output
    acc = tl.zeros((pad_q_tile_size, pad_head_dim), dtype=tl.float32)

    if is_causal:
        k_end = tl.minimum((q_tile_id + 1) * q_tile_size, seq_len)
    else:
        k_end = seq_len

    # loop over k/v tiles
    for k_start in tl.range(0, k_end, k_tile_size):
        k_start = tl.multiple_of(k_start, k_tile_size)
        kv_idx = k_start + k_offsets

        # load k tile
        k_tile_offsets = b * k_stride_b + hk * k_stride_h + kv_idx[:, None] * k_stride_n + d_offsets[None, :]

        if EVEN_N and EVEN_HEADDIM:
            k_tile = tl.load(k_ptr + k_tile_offsets)
        elif EVEN_N:
            k_tile = tl.load(k_ptr + k_tile_offsets, mask=d_offsets[None, :] < head_dim, other=0.)
        elif EVEN_HEADDIM:
            k_tile = tl.load(k_ptr + k_tile_offsets, mask=kv_idx[:, None] < seq_len, other=0.)
        else:
            k_tile = tl.load(k_ptr + k_tile_offsets, mask=(kv_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim), other=0.)

        # s = q @ k^T
        s_tile = tl.dot(q_tile, tl.trans(k_tile, (1, 0))) * sm_scale

        if is_causal:
            s_tile = tl.where((kv_idx[None, :] <= q_idx[:, None]) & (kv_idx[None, :] < seq_len), s_tile, -float('inf'))
        else:
            s_tile = tl.where(kv_idx[None, :] < seq_len, s_tile, -float('inf'))

        m_new = tl.maximum(m, tl.max(s_tile, axis=1))
        p_tile = tl.exp(s_tile - m_new[:, None])

        # load v tile
        v_tile_offsets = b * v_stride_b + hk * v_stride_h + kv_idx[:, None] * v_stride_n + d_offsets[None, :]

        if EVEN_N and EVEN_HEADDIM:
            v_tile = tl.load(v_ptr + v_tile_offsets)
        elif EVEN_N:
            v_tile = tl.load(v_ptr + v_tile_offsets, mask=d_offsets[None, :] < head_dim, other=0.)
        elif EVEN_HEADDIM:
            v_tile = tl.load(v_ptr + v_tile_offsets, mask=kv_idx[:, None] < seq_len, other=0.)
        else:
            v_tile = tl.load(v_ptr + v_tile_offsets, mask=(kv_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim), other=0.)

        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p_tile, axis=1)

        p_tile = p_tile.to(v_tile.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_tile, v_tile)

        m = m_new
        l = l_new

    o_tile_offsets = b * o_stride_b + hq * o_stride_h + q_idx[:, None] * o_stride_n + d_offsets[None, :]
    o_tile = acc / l[:, None]

    if EVEN_M and EVEN_HEADDIM:
        tl.store(out_ptr + o_tile_offsets, o_tile)
    elif EVEN_M:
        tl.store(out_ptr + o_tile_offsets, o_tile, mask=d_offsets[None, :] < head_dim)
    elif EVEN_HEADDIM:
        tl.store(out_ptr + o_tile_offsets, o_tile, mask=q_idx[:, None] < seq_len)
    else:
        tl.store(out_ptr + o_tile_offsets, o_tile, mask=(q_idx[:, None] < seq_len) & (d_offsets[None, :] < head_dim))


flash_attention_v2_fwd_autotuned_fused_kernel = triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=['n_q_heads', 'n_kv_heads', 'seq_len', 'head_dim', 'is_causal'],
)(flash_attention_v2_fwd_fused_kernel)


def flash_attention_v2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    q = [B, Hq, Q, D]
    k = [B, Hk, K, D]
    v = [B, Hv, K, D]

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
    assert q.dim() == k.dim() == v.dim() == 4
    assert k.shape == v.shape

    out = torch.empty_like(q)

    B, Hq, N, D = q.shape
    Bk, Hk, Nk, Dk = k.shape

    assert B == Bk
    assert N == Nk
    assert D == Dk
    assert Hq % Hk == 0
    assert Hq >= Hk

    # TODO: tune tile size and consider SRAM size
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(D)
    n_q_tiles = triton.cdiv(N, BLOCK_M)
    grid = (n_q_tiles, B * Hq)
    num_warps = 4 if D <= 64 else 8

    flash_attention_v2_fwd_fused_kernel[grid](
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
        Hq,
        Hk,
        N,
        D,
        1.0 / math.sqrt(D),
        is_causal,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        num_warps=num_warps,
    )

    return out


def flash_attention_v2_fwd_autotuned(
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
    assert q.dim() == k.dim() == v.dim() == 4
    assert k.shape == v.shape

    out = torch.empty_like(q)

    B, Hq, N, D = q.shape
    Bk, Hk, Nk, Dk = k.shape

    assert B == Bk
    assert N == Nk
    assert D == Dk
    assert Hq % Hk == 0
    assert Hq >= Hk

    def grid(meta):
        n_q_tiles = triton.cdiv(N, meta['BLOCK_M'])
        return (n_q_tiles, B * Hq)

    flash_attention_v2_fwd_autotuned_fused_kernel[grid](
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
        Hq,
        Hk,
        N,
        D,
        1.0 / math.sqrt(D),
        is_causal,
        BLOCK_D=triton.next_power_of_2(D),
    )

    return out


def flash_attention_v2_gqa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    q = [B, Hq, Q, D]
    k = [B, Hk, K, D]
    v = [B, Hk, K, D]

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
    assert q.dim() == k.dim() == v.dim() == 4
    assert k.shape == v.shape

    out = torch.empty_like(q)

    B, Hq, N, D = q.shape
    Bk, Hk, Nk, Dk = k.shape

    assert B == Bk
    assert N == Nk
    assert D == Dk
    assert Hq % Hk == 0
    assert Hq >= Hk

    # TODO: tune tile size and consider SRAM size
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_D = triton.next_power_of_2(D)
    n_q_tiles = triton.cdiv(N, BLOCK_M)
    grid = (n_q_tiles, B * Hq)
    num_warps = 4 if D <= 64 else 8

    flash_attention_v2_fwd_fused_kernel[grid](
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
        Hq,
        Hk,
        N,
        D,
        1.0 / math.sqrt(D),
        is_causal,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        num_warps=num_warps,
    )

    return out
