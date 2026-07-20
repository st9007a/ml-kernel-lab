import pytest
import torch
import torch.nn.functional as F

from ml_kernel_lab.kernel import triton_kernel


def torch_gqa_attention(q, k, v, is_causal=False):
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)


def assert_attention_close(actual, expected, dtype):
    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('batch_size', 'n_q_heads', 'n_kv_heads', 'seq_len', 'head_dim'),
    [
        (1, 8, 2, 64, 64),
        (2, 16, 4, 77, 64),
        (1, 32, 8, 129, 128),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_match_torch_results(batch_size, n_q_heads, n_kv_heads, seq_len, head_dim, dtype):
    q = torch.randn((batch_size, n_q_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    k = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    v = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device='cuda')

    actual = triton_kernel.flash_attention_v1_gqa_fwd(q, k, v)
    expected = torch_gqa_attention(q, k, v)

    assert_attention_close(actual, expected, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('batch_size', 'n_q_heads', 'n_kv_heads', 'seq_len', 'head_dim'),
    [
        (1, 8, 2, 64, 64),
        (2, 16, 4, 77, 64),
        (1, 32, 8, 129, 128),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_causal_match_torch_results(batch_size, n_q_heads, n_kv_heads, seq_len, head_dim, dtype):
    q = torch.randn((batch_size, n_q_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    k = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device='cuda')
    v = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device='cuda')

    actual = triton_kernel.flash_attention_v1_gqa_fwd(q, k, v, is_causal=True)
    expected = torch_gqa_attention(q, k, v, is_causal=True)

    assert_attention_close(actual, expected, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_match_torch_results_non_contiguous_bhn(dtype):
    batch_size = 2
    seq_len = 77
    n_q_heads = 16
    n_kv_heads = 4
    head_dim = 64

    q = torch.randn((batch_size, seq_len, n_q_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    k = torch.randn((batch_size, seq_len, n_kv_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    v = torch.randn((batch_size, seq_len, n_kv_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)

    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    actual = triton_kernel.flash_attention_v1_gqa_fwd(q, k, v)
    expected = torch_gqa_attention(q, k, v)

    assert_attention_close(actual, expected, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_causal_match_torch_results_non_contiguous_bhn(dtype):
    batch_size = 2
    seq_len = 77
    n_q_heads = 16
    n_kv_heads = 4
    head_dim = 64

    q = torch.randn((batch_size, seq_len, n_q_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    k = torch.randn((batch_size, seq_len, n_kv_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)
    v = torch.randn((batch_size, seq_len, n_kv_heads, head_dim), dtype=dtype, device='cuda').transpose(1, 2)

    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    actual = triton_kernel.flash_attention_v1_gqa_fwd(q, k, v, is_causal=True)
    expected = torch_gqa_attention(q, k, v, is_causal=True)

    assert_attention_close(actual, expected, dtype)
