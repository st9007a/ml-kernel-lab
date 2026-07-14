import pytest
import torch

from ml_kernel_lab.kernel import triton_kernel


def torch_masked_softmax(x, attn_mask):
    x_masked = x.masked_fill(attn_mask != 0, -float('inf'))
    row_has_valid = torch.isfinite(x_masked).any(dim=-1, keepdim=True)

    y = torch.softmax(x_masked, dim=-1)
    return torch.where(row_has_valid, y, torch.zeros_like(y))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(1, 8, 1, 64), (2, 4, 8, 64), (1, 8, 17, 128)])
@pytest.mark.parametrize('mask_q_len', ['one', 'full'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_match_torch_results(shape, mask_q_len, dtype):
    x = torch.randn(*shape, dtype=dtype, device='cuda')
    batch_size, _, q_len, kv_len = shape
    mask_shape = (batch_size, 1, 1 if mask_q_len == 'one' else q_len, kv_len)
    attn_mask = torch.rand(mask_shape, device='cuda') < 0.35

    actual = triton_kernel.masked_softmax_fwd(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mask_q_len', ['one', 'full'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_fully_masked_row_returns_zero(mask_q_len, dtype):
    x = torch.randn((2, 4, 3, 16), dtype=dtype, device='cuda')
    mask_shape = (2, 1, 1 if mask_q_len == 'one' else 3, 16)
    attn_mask = torch.zeros(mask_shape, dtype=torch.bool, device='cuda')
    attn_mask[1, :, :, :] = True

    actual = triton_kernel.masked_softmax_fwd(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(1, 8, 1, 64), (2, 4, 8, 64), (1, 8, 17, 128)])
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
    x = torch.randn((1, 4, 3, 16), dtype=torch.float32, device='cuda')
    attn_mask = torch.zeros_like(x, dtype=torch.bool)
    attn_mask[:, :, 2, :] = True

    actual = triton_kernel.masked_softmax_fwd_v2(x, attn_mask)
    expected = torch_masked_softmax(x, attn_mask)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
