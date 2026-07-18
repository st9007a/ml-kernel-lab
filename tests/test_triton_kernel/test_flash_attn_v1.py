import pytest
import torch
import torch.nn.functional as F

from ml_kernel_lab.kernel import triton_kernel


def torch_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def torch_causal_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    'shape',
    [
        (1, 2, 64, 64),
        (2, 4, 77, 64),
        (1, 8, 129, 128),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_match_torch_results(shape, dtype):
    q = torch.randn(*shape, dtype=dtype, device='cuda')
    k = torch.randn(*shape, dtype=dtype, device='cuda')
    v = torch.randn(*shape, dtype=dtype, device='cuda')

    actual = triton_kernel.flash_attention_v1_fwd(q, k, v)
    expected = torch_attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    'shape',
    [
        (1, 2, 64, 64),
        (2, 4, 77, 64),
        (1, 8, 129, 128),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_causal_match_torch_results(shape, dtype):
    q = torch.randn(*shape, dtype=dtype, device='cuda')
    k = torch.randn(*shape, dtype=dtype, device='cuda')
    v = torch.randn(*shape, dtype=dtype, device='cuda')

    actual = triton_kernel.flash_attention_v1_fwd(q, k, v, is_causal=True)
    expected = torch_causal_attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_match_torch_results_non_contiguous_bhn(dtype):
    batch_size = 2
    seq_len = 77
    n_heads = 4
    head_dim = 64

    q = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    k = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    v = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)

    assert q.stride(-1) == 1
    assert not q.is_contiguous()

    actual = triton_kernel.flash_attention_v1_fwd(q, k, v)
    expected = torch_attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_causal_match_torch_results_non_contiguous_bhn(dtype):
    batch_size = 2
    seq_len = 77
    n_heads = 4
    head_dim = 64

    q = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    k = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    v = torch.randn((batch_size, seq_len, n_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)

    assert q.stride(-1) == 1
    assert not q.is_contiguous()

    actual = triton_kernel.flash_attention_v1_fwd(q, k, v, is_causal=True)
    expected = torch_causal_attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
