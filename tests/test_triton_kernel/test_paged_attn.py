import pytest
import torch
import torch.nn.functional as F

from ml_kernel_lab.kernel import triton_kernel


def make_paged_cache(k, v, seq_lens, block_size, permute_blocks=False):
    batch_size, n_heads, max_seq_len, head_dim = k.shape
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_blocks_per_seq

    if permute_blocks:
        physical_blocks = torch.randperm(total_blocks, device=k.device, dtype=torch.int32)
    else:
        physical_blocks = torch.arange(total_blocks, device=k.device, dtype=torch.int32)

    block_table = physical_blocks.view(batch_size, max_blocks_per_seq)
    k_cache = torch.randn((total_blocks, n_heads, block_size, head_dim), dtype=k.dtype, device=k.device)
    v_cache = torch.randn_like(k_cache)

    for b in range(batch_size):
        for logical_block_idx in range(max_blocks_per_seq):
            start = logical_block_idx * block_size
            end = min(start + block_size, max_seq_len)
            physical_block_idx = int(block_table[b, logical_block_idx].item())
            k_cache[physical_block_idx, :, : end - start, :] = k[b, :, start:end, :]
            v_cache[physical_block_idx, :, : end - start, :] = v[b, :, start:end, :]

    return k_cache, v_cache, block_table, seq_lens.to(torch.int32)


def torch_decode_attention(q, k, v, seq_lens):
    outputs = []
    for b in range(q.shape[0]):
        seq_len = int(seq_lens[b].item())
        out = F.scaled_dot_product_attention(
            q[b : b + 1, :, None, :],
            k[b : b + 1, :, :seq_len, :],
            v[b : b + 1, :, :seq_len, :],
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(out[:, :, 0, :])
    return torch.cat(outputs, dim=0)


def assert_attention_close(actual, expected, dtype):
    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('batch_size', 'n_heads', 'max_seq_len', 'head_dim', 'block_size', 'seq_lens_values'),
    [
        (1, 4, 64, 64, 16, [64]),
        (2, 8, 77, 64, 16, [77, 53]),
        (3, 8, 129, 128, 32, [129, 97, 1]),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('permute_blocks', [False, True])
@pytest.mark.parametrize('impl', ['v1', 'v2'])
def test_match_torch_results(
    batch_size,
    n_heads,
    max_seq_len,
    head_dim,
    block_size,
    seq_lens_values,
    dtype,
    permute_blocks,
    impl,
):
    q = torch.randn((batch_size, n_heads, head_dim), dtype=dtype, device='cuda')
    k = torch.randn((batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device='cuda')
    v = torch.randn((batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device='cuda')
    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device='cuda')
    k_cache, v_cache, block_table, seq_lens = make_paged_cache(k, v, seq_lens, block_size, permute_blocks)
    max_num_blocks = (max(seq_lens_values) + block_size - 1) // block_size

    if impl == 'v1':
        actual = triton_kernel.single_query_paged_kv_attention(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)
    elif impl == 'v2':
        actual = triton_kernel.single_query_paged_kv_attention_v2(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)
    else:
        raise ValueError(f'unknown impl: {impl}')

    expected = torch_decode_attention(q, k, v, seq_lens)

    assert_attention_close(actual, expected, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_v2_match_torch_with_ragged_sequences_and_permuted_blocks():
    batch_size = 4
    n_heads = 8
    max_seq_len = 257
    head_dim = 128
    block_size = 32
    dtype = torch.bfloat16

    q = torch.randn((batch_size, n_heads, head_dim), dtype=dtype, device='cuda')
    k = torch.randn((batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device='cuda')
    v = torch.randn((batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device='cuda')
    seq_lens_values = [257, 193, 65, 1]
    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32, device='cuda')
    k_cache, v_cache, block_table, seq_lens = make_paged_cache(k, v, seq_lens, block_size, permute_blocks=True)
    max_num_blocks = (max(seq_lens_values) + block_size - 1) // block_size

    actual = triton_kernel.single_query_paged_kv_attention_v2(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)
    expected = torch_decode_attention(q, k, v, seq_lens)

    assert_attention_close(actual, expected, dtype)
