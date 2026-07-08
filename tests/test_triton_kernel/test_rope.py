import pytest
import torch

from ml_kernel_lab.kernel import triton_kernel


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def expand_cos_sin(cos, sin, batch_size, n_heads, seq_len, head_dim):
    if cos.shape[-1] == head_dim // 2:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

    if cos.shape[0] == 1:
        cos = cos.expand(batch_size, -1, -1)
        sin = sin.expand(batch_size, -1, -1)

    cos = cos[:, None, :, :].expand(batch_size, n_heads, seq_len, head_dim)
    sin = sin[:, None, :, :].expand(batch_size, n_heads, seq_len, head_dim)
    return cos, sin


def torch_rope(x, cos, sin):
    batch_size, n_heads, seq_len, head_dim = x.shape
    cos, sin = expand_cos_sin(cos, sin, batch_size, n_heads, seq_len, head_dim)
    return x * cos + rotate_half(x) * sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    'batch_size,n_q_head,n_k_head,seq_len,head_dim',
    [
        (1, 8, 8, 4, 64),
        (2, 12, 4, 7, 64),
        (2, 16, 8, 11, 128),
    ],
)
@pytest.mark.parametrize('cos_batch_size', [1, 2])
@pytest.mark.parametrize('cos_head_dim', ['half', 'full'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_match_torch_results(batch_size, n_q_head, n_k_head, seq_len, head_dim, cos_batch_size, cos_head_dim, dtype):
    if cos_batch_size != 1 and cos_batch_size != batch_size:
        pytest.skip('per-batch cos/sin must match batch size')

    q = torch.randn((batch_size, n_q_head, seq_len, head_dim), dtype=dtype, device='cuda')
    k = torch.randn((batch_size, n_k_head, seq_len, head_dim), dtype=dtype, device='cuda')

    angles = torch.randn((cos_batch_size, seq_len, head_dim // 2), dtype=torch.float32, device='cuda')
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    if cos_head_dim == 'full':
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

    actual_q, actual_k = triton_kernel.rope_fwd(q, k, cos, sin)
    expected_q = torch_rope(q, cos, sin)
    expected_k = torch_rope(k, cos, sin)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-2, atol=1e-2)
