import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def moe_grouped_expert_gemm_fwd_v1_fused_kernel(
    x_grouped_ptr,
    w_ptr,
    expert_offsets_ptr,
    out_ptr,
    x_stride_row: tl.constexpr,
    w_stride_e: tl.constexpr,
    w_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    EXPERT_CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M_tile = [expert_start : expert_start+BLOCK_SIZE_M, ]
    """
    max_m_tiles = (EXPERT_CAPACITY + BLOCK_M - 1) // BLOCK_M
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    expert_id = pid_x // max_m_tiles
    m_tile_id = pid_x % max_m_tiles
    n_tile_id = pid_y

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)

    m_start = expert_start + m_tile_id * BLOCK_M

    if m_start >= expert_end:
        return

    n_start = n_tile_id * BLOCK_N
    n_start = tl.multiple_of(n_start, 16)

    rows = m_start + tl.arange(0, BLOCK_M)
    cols = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    m_mask = rows < expert_end
    n_mask = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_start = tl.multiple_of(k_start, 16)
        k_start_offsets = k_start + k_offsets
        k_mask = k_start_offsets < K
        m_tile_offsets = rows[:, None] * x_stride_row + k_start_offsets[None, :]
        m_tile = tl.load(x_grouped_ptr + m_tile_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.)

        n_tile_offsets = expert_id * w_stride_e + k_start_offsets[:, None] * w_stride_row + cols[None, :]
        n_tile = tl.load(w_ptr + n_tile_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.)

        acc = tl.dot(m_tile, n_tile, acc=acc)

    out_offsets = rows[:, None] * out_stride_row + cols[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def moe_grouped_expert_gemm_fwd_v2_fused_kernel(
    x_grouped_ptr,
    w_ptr,
    expert_offsets_ptr,
    out_ptr,
    x_stride_row: tl.constexpr,
    w_stride_e: tl.constexpr,
    w_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    EXPERT_CAPACITY: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M_tile = [expert_start : expert_start+BLOCK_SIZE_M, ]
    """
    max_m_tiles = (EXPERT_CAPACITY + BLOCK_M - 1) // BLOCK_M
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    expert_id = pid_y
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_in_group = GROUP_SIZE_M * num_n_tiles
    group_id = pid_x // num_pid_in_group

    first_m = group_id * GROUP_SIZE_M
    group_size_m = min(max_m_tiles - first_m, GROUP_SIZE_M)

    pid_in_group = pid_x % num_pid_in_group
    m_tile_id = first_m + pid_in_group % group_size_m
    n_tile_id = pid_in_group // group_size_m

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)

    m_start = expert_start + m_tile_id * BLOCK_M

    if m_start >= expert_end:
        return

    n_start = n_tile_id * BLOCK_N
    n_start = tl.multiple_of(n_start, 16)

    rows = m_start + tl.arange(0, BLOCK_M)
    cols = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    m_mask = rows < expert_end
    n_mask = cols < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_start = tl.multiple_of(k_start, 16)
        k_start_offsets = k_start + k_offsets
        k_mask = k_start_offsets < K
        m_tile_offsets = rows[:, None] * x_stride_row + k_start_offsets[None, :]
        m_tile = tl.load(x_grouped_ptr + m_tile_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.)

        n_tile_offsets = expert_id * w_stride_e + k_start_offsets[:, None] * w_stride_row + cols[None, :]
        n_tile = tl.load(w_ptr + n_tile_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.)

        acc = tl.dot(m_tile, n_tile, acc=acc)

    out_offsets = rows[:, None] * out_stride_row + cols[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def moe_grouped_expert_gemm_fwd_v3_fused_kernel(
    x_grouped_ptr,
    w_ptr,
    expert_offsets_ptr,
    out_ptr,
    x_stride_row: tl.constexpr,
    w_stride_e: tl.constexpr,
    w_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    N_EXPERTS_PAD: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    tile_id = pid
    total_tiles = 0

    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles_in_group = GROUP_SIZE_M * num_n_tiles

    expert_offsets_cols = tl.arange(0, N_EXPERTS_PAD)
    expert_offsets = tl.load(expert_offsets_ptr + expert_offsets_cols, mask=expert_offsets_cols < N_EXPERTS + 1, other=0)
    expert_id = 0
    m_tiles_acc = tl.zeros((N_EXPERTS_PAD,), dtype=tl.int32)

    for i in tl.range(1, N_EXPERTS + 1):
        expert_m = expert_offsets[i] - expert_offsets[i - 1]
        num_m_tiles = tl.cdiv(expert_m, BLOCK_M)
        total_tiles += num_m_tiles * num_n_tiles
        m_tiles_acc[i] = m_tiles_acc[i - 1] + num_m_tiles
        expert_id = tl.where(tile_id >= total_tiles, i, expert_id)

    while tile_id < total_tiles:
        while tile_id >= m_tiles_acc[expert_id + 1] * num_n_tiles:
            expert_id += 1

        expert_tile_start = m_tiles_acc[expert_id] * num_n_tiles
        tile_offset = tile_id - expert_tile_start
        group_id = tile_offset // num_tiles_in_group
        first_m = group_id * GROUP_SIZE_M
        num_m_tiles = m_tiles_acc[expert_id + 1] - m_tiles_acc[expert_id]
        group_size_m = min(num_m_tiles - first_m, GROUP_SIZE_M)

        tile_in_group = tile_offset % num_tiles_in_group
        m_tile_id = first_m + tile_in_group % group_size_m
        n_tile_id = tile_in_group // group_size_m

        m_start = expert_offsets[expert_id] + m_tile_id * BLOCK_M
        n_start = n_tile_id * BLOCK_N

        rows = m_start + tl.arange(0, BLOCK_M)
        cols = n_start + tl.arange(0, BLOCK_N)

        k_offsets = tl.arange(0, BLOCK_K)
        m_mask = rows < expert_offsets[expert_id + 1]
        n_mask = cols < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_start_offsets = k_start + k_offsets
            k_mask = k_start_offsets < K

            m_tile_offsets = rows[:, None] * x_stride_row + k_start_offsets[None, :]
            m_tile = tl.load(x_grouped_ptr + m_tile_offsets, mask=m_mask[:, None] & k_mask[None, :], other=0.)

            n_tile_offsets = expert_id * w_stride_e + k_start_offsets[:, None] * w_stride_row + cols[None, :]
            n_tile = tl.load(w_ptr + n_tile_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.)

            acc = tl.dot(m_tile, n_tile, acc=acc)

        out_offsets = rows[:, None] * out_stride_row + cols[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])

        tile_id += step


moe_grouped_expert_gemm_fwd_v1_autotuned_fused_kernel = triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K', 'N_EXPERTS', 'EXPERT_CAPACITY'],
)(moe_grouped_expert_gemm_fwd_v1_fused_kernel)

moe_grouped_expert_gemm_fwd_v2_autotuned_fused_kernel = triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 16},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 16},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 16},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K', 'N_EXPERTS', 'EXPERT_CAPACITY'],
)(moe_grouped_expert_gemm_fwd_v2_fused_kernel)


def moe_grouped_expert_gemm_fwd_v1(
    x_grouped: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_capacity: int,
) -> torch.Tensor:
    """
    x_grouped = [B*T*top_k, D_model]
    w = [n_experts, D_model, D_ff]
    expert_offsets = [n_experts + 1]
    out = [B*T*top_k, D_ff]
    """
    M, K = x_grouped.shape
    n_experts, K_w, N = w.shape
    num_offsets = expert_offsets.numel()

    assert K == K_w
    assert num_offsets == n_experts + 1

    # Assume the following contiguous() calls do no-op.
    x_grouped = x_grouped.contiguous()
    w = w.contiguous()
    expert_offsets = expert_offsets.contiguous()

    out = torch.empty((M, N), dtype=x_grouped.dtype, device=x_grouped.device)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    MAX_M_TILES = triton.cdiv(expert_capacity, BLOCK_M)

    grid = (n_experts * MAX_M_TILES, triton.cdiv(N, BLOCK_N))
    num_warps = 4

    moe_grouped_expert_gemm_fwd_v1_fused_kernel[grid](
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        M,
        N,
        K,
        n_experts,
        expert_capacity,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
    )
    return out


def moe_grouped_expert_gemm_fwd_v1_autotuned(
    x_grouped: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_capacity: int,
) -> torch.Tensor:
    """
    x_grouped = [B*T*top_k, D_model]
    w = [n_experts, D_model, D_ff]
    expert_offsets = [n_experts + 1]
    out = [B*T*top_k, D_ff]
    """
    M, K = x_grouped.shape
    n_experts, K_w, N = w.shape
    num_offsets = expert_offsets.numel()

    assert K == K_w
    assert num_offsets == n_experts + 1

    # Assume the following contiguous() calls do no-op.
    x_grouped = x_grouped.contiguous()
    w = w.contiguous()
    expert_offsets = expert_offsets.contiguous()

    out = torch.empty((M, N), dtype=x_grouped.dtype, device=x_grouped.device)

    def grid(meta):
        max_m_tiles = triton.cdiv(expert_capacity, meta['BLOCK_M'])
        return (n_experts * max_m_tiles, triton.cdiv(N, meta['BLOCK_N']))

    moe_grouped_expert_gemm_fwd_v1_autotuned_fused_kernel[grid](
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        M,
        N,
        K,
        n_experts,
        expert_capacity,
    )

    return out


def moe_grouped_expert_gemm_fwd_v2(
    x_grouped: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_capacity: int,
) -> torch.Tensor:
    """
    x_grouped = [B*T*top_k, D_model]
    w = [n_experts, D_model, D_ff]
    expert_offsets = [n_experts + 1]
    out = [B*T*top_k, D_ff]
    """
    M, K = x_grouped.shape
    n_experts, K_w, N = w.shape
    num_offsets = expert_offsets.numel()

    assert K == K_w
    assert num_offsets == n_experts + 1

    # Assume the following contiguous() calls do no-op.
    x_grouped = x_grouped.contiguous()
    w = w.contiguous()
    expert_offsets = expert_offsets.contiguous()

    out = torch.empty((M, N), dtype=x_grouped.dtype, device=x_grouped.device)
    GROUP_SIZE_M = 8
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    MAX_M_TILES = triton.cdiv(expert_capacity, BLOCK_M)

    grid = (MAX_M_TILES * triton.cdiv(N, BLOCK_N), n_experts)
    num_warps = 4

    moe_grouped_expert_gemm_fwd_v2_fused_kernel[grid](
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        M,
        N,
        K,
        n_experts,
        expert_capacity,
        GROUP_SIZE_M,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
    )
    return out


def moe_grouped_expert_gemm_fwd_v2_autotuned(
    x_grouped: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_capacity: int,
) -> torch.Tensor:
    """
    x_grouped = [B*T*top_k, D_model]
    w = [n_experts, D_model, D_ff]
    expert_offsets = [n_experts + 1]
    out = [B*T*top_k, D_ff]
    """
    M, K = x_grouped.shape
    n_experts, K_w, N = w.shape
    num_offsets = expert_offsets.numel()

    assert K == K_w
    assert num_offsets == n_experts + 1

    # Assume the following contiguous() calls do no-op.
    x_grouped = x_grouped.contiguous()
    w = w.contiguous()
    expert_offsets = expert_offsets.contiguous()

    out = torch.empty((M, N), dtype=x_grouped.dtype, device=x_grouped.device)

    def grid(meta):
        max_m_tiles = triton.cdiv(expert_capacity, meta['BLOCK_M'])
        return (max_m_tiles * triton.cdiv(N, meta['BLOCK_N']), n_experts)

    moe_grouped_expert_gemm_fwd_v2_autotuned_fused_kernel[grid](
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        M,
        N,
        K,
        n_experts,
        expert_capacity,
    )

    return out


def moe_grouped_expert_gemm_fwd_v3(
    x_grouped: torch.Tensor,
    w: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_sms: int,
) -> torch.Tensor:
    """
    x_grouped = [B*T*top_k, D_model]
    w = [n_experts, D_model, D_ff]
    expert_offsets = [n_experts + 1]
    out = [B*T*top_k, D_ff]
    """
    M, K = x_grouped.shape
    n_experts, K_w, N = w.shape
    num_offsets = expert_offsets.numel()

    assert K == K_w
    assert num_offsets == n_experts + 1

    # Assume the following contiguous() calls do no-op.
    x_grouped = x_grouped.contiguous()
    w = w.contiguous()
    expert_offsets = expert_offsets.contiguous()

    out = torch.empty((M, N), dtype=x_grouped.dtype, device=x_grouped.device)
    GROUP_SIZE_M = 8
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    occupancy_multiplier = 1

    grid = (num_sms * occupancy_multiplier,)
    num_warps = 4

    moe_grouped_expert_gemm_fwd_v3_fused_kernel[grid](
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        M,
        N,
        K,
        n_experts,
        triton.next_power_of_2(n_experts + 1),
        GROUP_SIZE_M,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
    )
    return out
