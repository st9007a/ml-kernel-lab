import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_fused(
    x_ptr,
    gate_ptr,
    y_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implement simple Swish based gated linear unit:

    y = silu(gate) * x

    Assume the input tensors are contiguous.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    cols = offset + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements

    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
    gate = tl.load(gate_ptr + cols, mask=mask, other=0.).to(tl.float32)

    y = tl.sigmoid(gate) * gate * x
    tl.store(y_ptr + cols, y, mask=mask)


def swiglu_fwd(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    assert x.shape == gate.shape

    x = x.contiguous()
    gate = gate.contiguous()

    y = torch.empty_like(x)

    BLOCK_SIZE = 65536 // x.element_size()
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    grid = (triton.cdiv(y.numel(), BLOCK_SIZE), )

    _swiglu_fwd_fused[grid](x, gate, y, y.numel(), BLOCK_SIZE, num_warps=num_warps)

    return y
