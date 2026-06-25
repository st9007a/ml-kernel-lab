import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["N", "BLOCK_SIZE"],
)
@triton.jit
def _rms_norm_fwd_fused(
    x_ptr,
    y_ptr,
    w_ptr,
    stride: int,
    N: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr
):
    """
    X, Y are 2D tensors and W is a 1D tensor.

    Args:
      - x_ptr: pointer to input
      - y_ptr: pointer to output
      - w_ptr: pointer to weights
      - stride: the total steps to move the input pointer by 1 row
      - N: number of columns
      - eps: epsilon to avoid division by zero
    """
    row = tl.program_id(0)

    x_ptr += row * stride
    y_ptr += row * stride

    tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        tmp += x * x

    inv_rms = tl.rsqrt(tl.sum(tmp, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        y = x * inv_rms * w

        tl.store(y_ptr + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["N", "BLOCK_SIZE"],
)
@triton.jit
def _rms_norm_fwd_fused_v2(
    x_ptr,
    y_ptr,
    w_ptr,
    stride: int,
    N: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr
):
    """
    X, Y are 2D tensors and W is a 1D tensor.

    Args:
      - x_ptr: pointer to input
      - y_ptr: pointer to output
      - w_ptr: pointer to weights
      - stride: the total steps to move the input pointer by 1 row
      - N: number of columns
      - eps: epsilon to avoid division by zero
    """
    row = tl.program_id(0)

    x_ptr += row * stride
    y_ptr += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)

    w = tl.load(w_ptr + cols, mask=mask)
    y = x * inv_rms * w

    tl.store(y_ptr + cols, y, mask=mask)


def rms_norm_fwd(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.contiguous()

    y = torch.empty_like(x)
    x = x.reshape(-1, x.shape[-1])

    M, N = x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    assert N <= BLOCK_SIZE, "feature dim more than 64KB is not yet supported"

    grid = (M,)
    # num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _rms_norm_fwd_fused[grid](x, y, w, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_ctas=1)

    return y


def rms_norm_fwd_v2(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.contiguous()

    y = torch.empty_like(x)
    x = x.reshape(-1, x.shape[-1])

    M, N = x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    assert N <= BLOCK_SIZE, "feature dim more than 64KB is not yet supported"

    grid = (M,)
    # num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _rms_norm_fwd_fused_v2[grid](x, y, w, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_ctas=1)

    return y
