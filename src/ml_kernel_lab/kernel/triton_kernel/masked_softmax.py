import torch
import triton
import triton.language as tl


@triton.jit
def masked_softmax_fwd_fused_kernel(
    x_ptr,
    y_ptr,
    attn_mask_ptr,
    n_h: tl.constexpr,
    n_q: tl.constexpr,
    n_k: tl.constexpr,
    n_attn_mask_q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    batch_id = row // (n_h * n_q)
    cols = tl.arange(0, BLOCK_SIZE)
    bounds_mask = cols < n_k

    if n_attn_mask_q == 1:
        attn_mask_offsets = batch_id * n_attn_mask_q * n_k
    else:
        q = row % n_q
        attn_mask_offsets = batch_id * n_attn_mask_q * n_k + q * n_k

    x = tl.load(x_ptr + row * n_k + cols, mask=bounds_mask, other=-float('inf'))
    attn_mask = tl.load(attn_mask_ptr + attn_mask_offsets + cols, mask=bounds_mask, other=1.)

    x_masked = tl.where(attn_mask != 0, -float('inf'), x)
    x_max = tl.max(x_masked, axis=0)

    row_has_valid = x_max != -float('inf')

    x_shifted = tl.where(row_has_valid, x_masked - x_max, 0.)

    numerator = tl.where(row_has_valid, tl.exp(x_shifted.to(tl.float32)), 0.)
    denominator = tl.where(row_has_valid, tl.sum(numerator, axis=0), 1.)

    y = numerator / denominator
    tl.store(y_ptr + row * n_k + cols, y, mask=bounds_mask)


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


@triton.jit
def masked_softmax_fwd_fused_kernel_v3(
    x_ptr,
    y_ptr,
    attn_mask_ptr,
    n_h: tl.constexpr,
    n_q: tl.constexpr,
    n_k: tl.constexpr,
    n_attn_mask_q: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bq = tl.program_id(0)
    pid_h = tl.program_id(1)

    batch_id = pid_bq // n_q
    q = pid_bq % n_q
    h = pid_h * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    cols = tl.arange(0, BLOCK_SIZE)

    h_mask = h < n_h
    bounds_mask = cols < n_k

    x_offsets = (
        batch_id * n_h * n_q * n_k
        + h[:, None] * n_q * n_k
        + q * n_k
        + cols[None, :]
    )

    if n_attn_mask_q == 1:
        attn_mask_offsets = batch_id * n_attn_mask_q * n_k
    else:
        attn_mask_offsets = batch_id * n_attn_mask_q * n_k + q * n_k

    x = tl.load(x_ptr + x_offsets, mask=h_mask[:, None] & bounds_mask[None, :], other=-float('inf'))
    attn_mask = tl.load(attn_mask_ptr + attn_mask_offsets + cols, mask=bounds_mask, other=1.)

    x_masked = tl.where(attn_mask[None, :] != 0, -float('inf'), x)
    x_max = tl.max(x_masked, axis=1)[:, None]

    row_has_valid = x_max != -float('inf')
    x_shifted = tl.where(row_has_valid, x_masked - x_max, 0.)

    numerator = tl.where(row_has_valid, tl.exp(x_shifted.to(tl.float32)), 0.)
    denominator = tl.where(row_has_valid, tl.sum(numerator, axis=1)[:, None], 1.)

    y = numerator / denominator
    tl.store(y_ptr + x_offsets, y, mask=h_mask[:, None] & bounds_mask[None, :])


def masked_softmax_fwd(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    x = [B, H, Q, K]
    attn_mask = [B, 1, Q, K] or [B, 1, 1, K]
    y = [B, H, Q, K]
    """
    B, H, Q, K = x.shape
    mB, mH, mQ, mK = attn_mask.shape

    assert B == mB
    assert mH == 1
    assert Q == mQ or mQ == 1
    assert K == mK

    x = x.contiguous()
    attn_mask = attn_mask.contiguous()
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(K)
    grid = (B * H * Q, )
    num_warps = 4
    masked_softmax_fwd_fused_kernel[grid](
        x,
        y,
        attn_mask,
        H,
        Q,
        K,
        mQ,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y


def masked_softmax_fwd_v2(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    assert x.shape == attn_mask.shape

    head_dim = x.shape[-1]

    x_2d = x.view(-1, head_dim)
    attn_mask_2d = attn_mask.view(-1, head_dim)
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


def masked_softmax_fwd_v3(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    x = [B, H, Q, K]
    attn_mask = [B, 1, Q, K] or [B, 1, 1, K]
    y = [B, H, Q, K]
    """
    B, H, Q, K = x.shape
    mB, mH, mQ, mK = attn_mask.shape

    assert B == mB
    assert mH == 1
    assert Q == mQ or mQ == 1
    assert K == mK

    x = x.contiguous()
    attn_mask = attn_mask.contiguous()
    y = torch.empty_like(x)

    HEAD_BLOCK = 4
    BLOCK_SIZE = triton.next_power_of_2(K)
    grid = (B * Q, triton.cdiv(H, HEAD_BLOCK))
    num_warps = 4
    masked_softmax_fwd_fused_kernel_v3[grid](
        x,
        y,
        attn_mask,
        H,
        Q,
        K,
        mQ,
        HEAD_BLOCK,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y
