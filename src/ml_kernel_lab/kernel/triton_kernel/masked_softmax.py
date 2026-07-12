import torch
import triton
import triton.language as tl


@triton.jit
def masked_softmax_fwd_fused_kernel(
    x_ptr,
    y_ptr,
    attn_mask_ptr,
    x_row_stride,
    y_row_stride,
    attn_mask_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    bounds_mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=bounds_mask, other=-float('inf'))
    attn_mask = tl.load(attn_mask_ptr + row * attn_mask_row_stride + cols, mask=bounds_mask, other=1.)

    x_masked = tl.where(attn_mask != 0, -float('inf'), x)
    x_max = tl.max(x_masked, axis=0)

    row_has_valid = x_max != -float('inf')

    x_shifted = tl.where(row_has_valid, x_masked - x_max, 0.)

    numerator = tl.where(row_has_valid, tl.exp(x_shifted.to(tl.float32)), 0.)
    denominator = tl.where(row_has_valid, tl.sum(numerator, axis=0), 1.)

    y = numerator / denominator
    tl.store(y_ptr + row * y_row_stride + cols, y, mask=bounds_mask)


@triton.jit
def masked_softmax_fwd_fused_kernel_v2(
    x_ptr,
    y_ptr,
    attn_mask_ptr,
    x_row_stride,
    y_row_stride,
    attn_mask_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    bounds_mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=bounds_mask, other=-float('inf'))
    attn_mask = tl.load(attn_mask_ptr + row * attn_mask_row_stride + cols, mask=bounds_mask, other=1.)

    x_masked = tl.where(attn_mask != 0, -float('inf'), x)
    y = x_masked.softmax()
    tl.store(y_ptr + row * y_row_stride + cols, y, mask=bounds_mask)


def masked_softmax_fwd(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    assert x.shape == attn_mask.shape

    x_2d = x.reshape(-1, x.shape[-1])
    attn_mask_2d = attn_mask.reshape(-1, attn_mask.shape[-1])
    y_2d = torch.empty_like(x_2d)

    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )
    num_warps = 4
    masked_softmax_fwd_fused_kernel[grid](
        x_2d,
        y_2d,
        attn_mask_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        attn_mask_2d.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y_2d.reshape(x.shape)


def masked_softmax_fwd_v2(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    assert x.shape == attn_mask.shape

    x_2d = x.reshape(-1, x.shape[-1])
    attn_mask_2d = attn_mask.reshape(-1, attn_mask.shape[-1])
    y_2d = torch.empty_like(x_2d)

    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )
    num_warps = 4
    masked_softmax_fwd_fused_kernel_v2[grid](
        x_2d,
        y_2d,
        attn_mask_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        attn_mask_2d.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y_2d.reshape(x.shape)
