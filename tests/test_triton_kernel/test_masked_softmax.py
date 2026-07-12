import pytest
import torch

from ml_kernel_lab.kernel import triton_kernel


def torch_masked_softmax(x, attn_mask):
    x_masked = x.masked_fill(attn_mask != 0, -float('inf'))
    row_has_valid = torch.isfinite(x_masked).any(dim=-1, keepdim=True)

    y = torch.softmax(x_masked, dim=-1)
    return torch.where(row_has_valid, y, torch.zeros_like(y))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(4, 16), (2, 3, 32), (2, 4, 8, 64), (1, 8, 17, 128)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_match_torch_results(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device='cuda')
    attn_mask = torch.rand(*shape, device='cuda') < 0.35

    actual = triton_kernel.masked_softmax_fwd(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_fully_masked_row_returns_zero(dtype):
    x = torch.randn((4, 16), dtype=dtype, device='cuda')
    attn_mask = torch.zeros_like(x, dtype=torch.bool)
    attn_mask[1, :] = True
    attn_mask[3, :] = True

    actual = triton_kernel.masked_softmax_fwd(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)
    torch.testing.assert_close(actual[3], torch.zeros_like(actual[3]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(4, 16), (2, 3, 32), (2, 4, 8, 64), (1, 8, 17, 128)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_v2_match_torch_results_without_fully_masked_rows(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device='cuda')
    attn_mask = torch.rand(*shape, device='cuda') < 0.35
    attn_mask[..., 0] = False

    actual = triton_kernel.masked_softmax_fwd_v2(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.xfail(reason='v2 intentionally does not handle fully masked rows')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_v2_fully_masked_row_returns_zero():
    x = torch.randn((4, 16), dtype=torch.float32, device='cuda')
    attn_mask = torch.zeros_like(x, dtype=torch.bool)
    attn_mask[2, :] = True

    actual = triton_kernel.masked_softmax_fwd_v2(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
