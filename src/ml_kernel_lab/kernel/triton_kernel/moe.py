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
    MAX_M_TILES: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M_tile = [expert_start : expert_start+BLOCK_SIZE_M, ]
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    expert_id = pid_x // MAX_M_TILES
    m_tile_id = pid_x % MAX_M_TILES
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

        acc += tl.dot(m_tile, n_tile)

    out_offsets = rows[:, None] * out_stride_row + cols[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


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
        MAX_M_TILES,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
    )
    return out
