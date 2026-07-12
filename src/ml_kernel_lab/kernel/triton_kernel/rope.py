import torch
import triton
import triton.language as tl


@triton.jit
def rope_fwd_fused_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    batch_size: tl.constexpr,
    cos_batch_size: tl.constexpr,
    n_q_head: tl.constexpr,
    n_k_head: tl.constexpr,
    head_dim: tl.constexpr,
    pad_n_q_head: tl.constexpr,
    pad_n_k_head: tl.constexpr,
    pad_head_dim: tl.constexpr,
):
    """
    q = B, S, Nq, D
    k = B, S, Nk, D
    cos = B/1, S, D
    sin = B/1, S, D
    """
    pid = tl.program_id(0)

    q_ptr += pid * q_row_stride
    k_ptr += pid * k_row_stride

    batch_id = pid // seq_len
    seq_id = pid % seq_len

    cos_ptr += tl.where(
        cos_batch_size == 1,
        seq_id * cos_row_stride,
        batch_id * seq_len * cos_row_stride + seq_id * cos_row_stride,
    )
    sin_ptr += tl.where(
        cos_batch_size == 1,
        seq_id * sin_row_stride,
        batch_id * seq_len * sin_row_stride + seq_id * sin_row_stride,
    )

    cos_row_offset = tl.arange(0, pad_head_dim // 2)
    cos_row_mask = cos_row_offset < head_dim // 2
    cos_row = tl.load(cos_ptr + cos_row_offset, mask=cos_row_mask, other=0)
    sin_row = tl.load(sin_ptr + cos_row_offset, mask=cos_row_mask, other=0)

    first_half_q_offset = tl.arange(0, pad_n_q_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]
    first_half_k_offset = tl.arange(0, pad_n_k_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]

    first_half_q_mask = (tl.arange(0, pad_n_q_head)[:, None] < n_q_head) & (tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2)
    first_half_k_mask = (tl.arange(0, pad_n_k_head)[:, None] < n_k_head) & (tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offset, mask=first_half_q_mask, other=0)
    k_tile_1 = tl.load(k_ptr + first_half_k_offset, mask=first_half_k_mask, other=0)

    second_half_q_offset = first_half_q_offset + head_dim // 2
    second_half_k_offset = first_half_k_offset + head_dim // 2
    second_half_q_mask = first_half_q_mask
    second_half_k_mask = first_half_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offset, mask=second_half_q_mask, other=0)
    k_tile_2 = tl.load(k_ptr + second_half_k_offset, mask=second_half_k_mask, other=0)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offset, new_q_tile_1, mask=first_half_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offset, new_q_tile_2, mask=second_half_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offset, new_k_tile_1, mask=first_half_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offset, new_k_tile_2, mask=second_half_k_mask)


def rope_fwd(q, k, cos, sin):
    """
    q: [batch_size, n_q_head, seq_len, head_dim]
    k: [batch_size, n_k_head, seq_len, head_dim]

    Note that q, k are non-contiguous. Their layout is [batch_size, seq_len, n_head, head_dim]
    """

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_k_head = k.shape[2]
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_k_head = triton.next_power_of_2(n_k_head)
    pad_head_dim = triton.next_power_of_2(head_dim)
    cos_batch_size = cos.shape[0]

    n_rows = batch_size * seq_len

    rope_fwd_fused_kernel[(n_rows, )](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(1),
        sin,
        sin.stride(1),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_k_head,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_head_dim,
    )

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    return q, k
