import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


def make_paged_cache(k, v, block_size, permute_blocks=False):
    batch_size, n_kv_heads, max_seq_len, head_dim = k.shape
    max_blocks_per_seq = triton.cdiv(max_seq_len, block_size)
    total_blocks = batch_size * max_blocks_per_seq

    if permute_blocks:
        physical_blocks = torch.randperm(total_blocks, device=k.device, dtype=torch.int32)
    else:
        physical_blocks = torch.arange(total_blocks, device=k.device, dtype=torch.int32)

    block_table = physical_blocks.view(batch_size, max_blocks_per_seq)
    k_cache = torch.empty((total_blocks, n_kv_heads, block_size, head_dim), dtype=k.dtype, device=k.device)
    v_cache = torch.empty_like(k_cache)

    for b in range(batch_size):
        for logical_block_idx in range(max_blocks_per_seq):
            start = logical_block_idx * block_size
            end = min(start + block_size, max_seq_len)
            physical_block_idx = int(block_table[b, logical_block_idx].item())
            k_cache[physical_block_idx, :, : end - start, :] = k[b, :, start:end, :]
            v_cache[physical_block_idx, :, : end - start, :] = v[b, :, start:end, :]

    return k_cache, v_cache, block_table


def torch_gqa_decode_attention(q, k, v, seq_lens):
    outputs = []
    for b in range(q.shape[0]):
        seq_len = int(seq_lens[b].item())
        out = F.scaled_dot_product_attention(
            q[b : b + 1, :, None, :],
            k[b : b + 1, :, :seq_len, :],
            v[b : b + 1, :, :seq_len, :],
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        outputs.append(out[:, :, 0, :])
    return torch.cat(outputs, dim=0)


def torch_paged_gqa_decode_attention(q, k_cache, v_cache, block_table, seq_lens):
    batch_size, n_query_heads, head_dim = q.shape
    _, n_kv_heads, block_size, _ = k_cache.shape
    max_blocks_per_seq = block_table.shape[1]
    max_seq_len = max_blocks_per_seq * block_size
    physical_blocks = block_table.to(torch.long)

    k = k_cache[physical_blocks].permute(0, 2, 1, 3, 4).reshape(batch_size, n_kv_heads, max_seq_len, head_dim)
    v = v_cache[physical_blocks].permute(0, 2, 1, 3, 4).reshape(batch_size, n_kv_heads, max_seq_len, head_dim)
    assert n_query_heads % n_kv_heads == 0
    return torch_gqa_decode_attention(q, k, v, seq_lens)


compiled_torch_gqa_decode_attention = torch.compile(torch_gqa_decode_attention)
compiled_torch_paged_gqa_decode_attention = torch.compile(torch_paged_gqa_decode_attention)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton-v1', 'torch-paged', 'torch-paged.compile', 'torch-dense', 'torch-dense.compile'],
        line_names=['Triton v1', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense', 'Torch Dense Compile'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('gray', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='paged-attn-gqa-decode-forward-latency-seq-len',
        args={
            'batch_size': 8,
            'n_query_heads': 32,
            'n_kv_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16, 32],
        line_arg='provider',
        line_vals=['triton-v1', 'torch-paged', 'torch-paged.compile', 'torch-dense', 'torch-dense.compile'],
        line_names=['Triton v1', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense', 'Torch Dense Compile'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('gray', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='paged-attn-gqa-decode-forward-latency-batch-size',
        args={
            'seq_len': 1024,
            'n_query_heads': 32,
            'n_kv_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['n_kv_heads'],
        x_vals=[4, 8, 16, 32],
        line_arg='provider',
        line_vals=['triton-v1', 'torch-paged', 'torch-dense'],
        line_names=['Triton v1', 'Torch Paged', 'Torch Dense'],
        styles=[('blue', '-'), ('green', '-'), ('gray', '-')],
        ylabel='ms',
        plot_name='paged-attn-gqa-decode-forward-latency-kv-heads',
        args={
            'batch_size': 8,
            'seq_len': 1024,
            'n_query_heads': 32,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['head_dim'],
        x_vals=[64, 128],
        line_arg='provider',
        line_vals=['triton-v1', 'torch-paged', 'torch-paged.compile', 'torch-dense', 'torch-dense.compile'],
        line_names=['Triton v1', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense', 'Torch Dense Compile'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('gray', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='paged-attn-gqa-decode-forward-latency-head-dim',
        args={
            'batch_size': 8,
            'seq_len': 1024,
            'n_query_heads': 32,
            'n_kv_heads': 8,
            'block_size': 16,
        },
    ),
])
def bench_paged_attn_gqa(
    batch_size,
    n_query_heads,
    n_kv_heads,
    seq_len,
    head_dim,
    block_size,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    q = torch.randn((batch_size, n_query_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, n_kv_heads, seq_len, head_dim), dtype=dtype, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    k_cache, v_cache, block_table = make_paged_cache(k, v, block_size, permute_blocks=True)
    max_num_blocks = triton.cdiv(seq_len, block_size)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton-v1':
            return triton_kernel.single_query_paged_kv_attention(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)

        if provider == 'torch-paged':
            return torch_paged_gqa_decode_attention(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'torch-paged.compile':
            return compiled_torch_paged_gqa_decode_attention(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'torch-dense':
            return torch_gqa_decode_attention(q, k, v, seq_lens)

        if provider == 'torch-dense.compile':
            return compiled_torch_gqa_decode_attention(q, k, v, seq_lens)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_paged_attn_gqa.run(print_data=True, return_df=True)
