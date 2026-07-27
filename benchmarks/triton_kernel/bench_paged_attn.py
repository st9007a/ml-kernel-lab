import torch
import triton

from ml_kernel_lab.kernel import triton_kernel


def make_paged_cache(k, v, block_size, permute_blocks=False):
    batch_size, n_heads, max_seq_len, head_dim = k.shape
    max_blocks_per_seq = triton.cdiv(max_seq_len, block_size)
    total_blocks = batch_size * max_blocks_per_seq

    if permute_blocks:
        physical_blocks = torch.randperm(total_blocks, device=k.device, dtype=torch.int32)
    else:
        physical_blocks = torch.arange(total_blocks, device=k.device, dtype=torch.int32)

    block_table = physical_blocks.view(batch_size, max_blocks_per_seq)
    k_cache = torch.empty((total_blocks, n_heads, block_size, head_dim), dtype=k.dtype, device=k.device)
    v_cache = torch.empty_like(k_cache)

    for b in range(batch_size):
        for logical_block_idx in range(max_blocks_per_seq):
            start = logical_block_idx * block_size
            end = min(start + block_size, max_seq_len)
            physical_block_idx = int(block_table[b, logical_block_idx].item())
            k_cache[physical_block_idx, :, : end - start, :] = k[b, :, start:end, :]
            v_cache[physical_block_idx, :, : end - start, :] = v[b, :, start:end, :]

    return k_cache, v_cache, block_table


def torch_decode_attention(q, k, v, seq_lens):
    scores = torch.einsum('bhd,bhnd->bhn', q, k) * (q.shape[-1] ** -0.5)
    token_idx = torch.arange(k.shape[2], device=k.device)
    mask = token_idx[None, :] < seq_lens[:, None]
    scores = scores.masked_fill(~mask[:, None, :], -float('inf'))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum('bhn,bhnd->bhd', probs, v)


compiled_torch_decode_attention = torch.compile(torch_decode_attention)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton-v1', 'triton-v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='paged-attn-decode-forward-latency-seq-len',
        args={
            'batch_size': 8,
            'n_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16, 32],
        line_arg='provider',
        line_vals=['triton-v1', 'triton-v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='paged-attn-decode-forward-latency-batch-size',
        args={
            'seq_len': 1024,
            'n_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['block_size'],
        x_vals=[8, 16, 32, 64],
        line_arg='provider',
        line_vals=['triton-v1', 'triton-v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='paged-attn-decode-forward-latency-block-size',
        args={
            'batch_size': 8,
            'seq_len': 1024,
            'n_heads': 8,
            'head_dim': 128,
        },
    ),
    triton.testing.Benchmark(
        x_names=['head_dim'],
        x_vals=[64, 128],
        line_arg='provider',
        line_vals=['triton-v1', 'triton-v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='paged-attn-decode-forward-latency-head-dim',
        args={
            'batch_size': 8,
            'seq_len': 1024,
            'n_heads': 8,
            'block_size': 16,
        },
    ),
])
def bench_paged_attn(
    batch_size,
    n_heads,
    seq_len,
    head_dim,
    block_size,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    q = torch.randn((batch_size, n_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    k_cache, v_cache, block_table = make_paged_cache(k, v, block_size, permute_blocks=True)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton-v1':
            return triton_kernel.single_query_paged_kv_attention(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'triton-v2':
            return triton_kernel.single_query_paged_kv_attention_v2(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'torch':
            return torch_decode_attention(q, k, v, seq_lens)

        if provider == 'torch.compile':
            return compiled_torch_decode_attention(q, k, v, seq_lens)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_paged_attn.run(print_data=True, return_df=True)
