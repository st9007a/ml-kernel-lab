import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    X, Y are 2D tensors and W is a 1D tensor.

    Args:
      - X: pointer to input
      - Y: pointer to output
      - W: pointer to weights
      - stride: the total steps to move the input pointer by 1 row
      - N: number of columns
      - eps: epsilon to avoid division by zero
    """
    row = tl.program_id(0)

    X += row * stride
    Y += row * stride

    tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        tmp += x * x

    inv_rms = tl.rsqrt(tl.sum(tmp, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = x * inv_rms * w

        tl.store(Y + cols, y, mask=mask)


def rms_norm_fwd(X: torch.Tensor, W: torch.Tensor, eps: float) -> torch.Tensor:
    X = X.contiguous()

    Y = torch.empty_like(X)
    X = X.reshape(-1, X.shape[-1])

    M, N = X.shape
    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    assert N <= BLOCK_SIZE, "feature dim more than 64KB is not yet supported"

    grid = (M,)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _rms_norm_fwd_fused[grid](X, Y, W, X.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)

    return Y
