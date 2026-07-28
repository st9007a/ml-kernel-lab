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

    logical_k_blocks = k.reshape(batch_size, n_heads, max_blocks_per_seq, block_size, head_dim)
    logical_v_blocks = v.reshape(batch_size, n_heads, max_blocks_per_seq, block_size, head_dim)
    logical_k_blocks = logical_k_blocks.permute(0, 2, 1, 3, 4).reshape(total_blocks, n_heads, block_size, head_dim)
    logical_v_blocks = logical_v_blocks.permute(0, 2, 1, 3, 4).reshape(total_blocks, n_heads, block_size, head_dim)
    k_cache[physical_blocks.to(torch.long)] = logical_k_blocks
    v_cache[physical_blocks.to(torch.long)] = logical_v_blocks

    return k_cache, v_cache, block_table


def torch_decode_attention(q, k, v, seq_lens):
    scores = torch.einsum('bhd,bhnd->bhn', q, k) * (q.shape[-1] ** -0.5)
    token_idx = torch.arange(k.shape[2], device=k.device)
    mask = token_idx[None, :] < seq_lens[:, None]
    scores = scores.masked_fill(~mask[:, None, :], -float('inf'))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum('bhn,bhnd->bhd', probs, v)


def torch_paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens):
    batch_size, n_heads, head_dim = q.shape
    _, _, block_size, _ = k_cache.shape
    max_blocks_per_seq = block_table.shape[1]
    max_seq_len = max_blocks_per_seq * block_size
    physical_blocks = block_table.to(torch.long)

    k = k_cache[physical_blocks].permute(0, 2, 1, 3, 4).reshape(batch_size, n_heads, max_seq_len, head_dim)
    v = v_cache[physical_blocks].permute(0, 2, 1, 3, 4).reshape(batch_size, n_heads, max_seq_len, head_dim)
    return torch_decode_attention(q, k, v, seq_lens)


compiled_torch_decode_attention = torch.compile(torch_decode_attention)
compiled_torch_paged_decode_attention = torch.compile(torch_paged_decode_attention)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton-v1', 'triton-v2', 'torch-paged', 'torch-paged.compile', 'torch-dense'],
        line_names=['Triton v1', 'Triton v2', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-'), ('gray', '-')],
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
        line_vals=['triton-v1', 'triton-v2', 'torch-paged', 'torch-paged.compile', 'torch-dense'],
        line_names=['Triton v1', 'Triton v2', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-'), ('gray', '-')],
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
        line_vals=['triton-v1', 'triton-v2', 'torch-paged', 'torch-paged.compile', 'torch-dense'],
        line_names=['Triton v1', 'Triton v2', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-'), ('gray', '-')],
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
        line_vals=['triton-v1', 'triton-v2', 'torch-paged', 'torch-paged.compile', 'torch-dense'],
        line_names=['Triton v1', 'Triton v2', 'Torch Paged', 'Torch Paged Compile', 'Torch Dense'],
        styles=[('blue', '-'), ('cyan', '-'), ('green', '-'), ('red', '-'), ('gray', '-')],
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
    max_num_blocks = triton.cdiv(seq_len, block_size)
    num_blocks_per_split = 32
    num_splits = triton.cdiv(max_num_blocks, num_blocks_per_split)
    v2_acc = torch.empty((batch_size * n_heads, num_splits, head_dim), dtype=torch.float32, device=device)
    v2_local_max = torch.empty((batch_size * n_heads, num_splits), dtype=torch.float32, device=device)
    v2_local_expsum = torch.empty_like(v2_local_max)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton-v1':
            return triton_kernel.single_query_paged_kv_attention(q, k_cache, v_cache, block_table, seq_lens, max_num_blocks)

        if provider == 'triton-v2':
            return triton_kernel.single_query_paged_kv_attention_v2(
                q,
                k_cache,
                v_cache,
                block_table,
                seq_lens,
                max_num_blocks,
                num_blocks_per_split,
                v2_acc,
                v2_local_max,
                v2_local_expsum,
            )

        if provider == 'torch-paged':
            return torch_paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'torch-paged.compile':
            return compiled_torch_paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens)

        if provider == 'torch-dense':
            return torch_decode_attention(q, k, v, seq_lens)

        if provider == 'torch-dense.compile':
            return compiled_torch_decode_attention(q, k, v, seq_lens)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16],
        line_arg='num_blocks_per_split',
        line_vals=[1, 2, 4, 8, 16, 32, 64],
        line_names=['1', '2', '4', '8', '16', '32', '64'],
        styles=[
            ('blue', '-'),
            ('cyan', '-'),
            ('green', '-'),
            ('red', '-'),
            ('purple', '-'),
            ('orange', '-'),
            ('gray', '-'),
        ],
        ylabel='ms',
        plot_name='paged-attn-v2-forward-latency-split-size-batch-size',
        args={
            'seq_len': 1024,
            'n_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='num_blocks_per_split',
        line_vals=[1, 2, 4, 8, 16, 32, 64],
        line_names=['1', '2', '4', '8', '16', '32', '64'],
        styles=[
            ('blue', '-'),
            ('cyan', '-'),
            ('green', '-'),
            ('red', '-'),
            ('purple', '-'),
            ('orange', '-'),
            ('gray', '-'),
        ],
        ylabel='ms',
        plot_name='paged-attn-v2-forward-latency-split-size-seq-len',
        args={
            'batch_size': 1,
            'n_heads': 8,
            'head_dim': 128,
            'block_size': 16,
        },
    ),
])
def bench_paged_attn_v2_split_size(
    batch_size,
    n_heads,
    seq_len,
    head_dim,
    block_size,
    num_blocks_per_split,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    q = torch.randn((batch_size, n_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    k_cache, v_cache, block_table = make_paged_cache(k, v, block_size, permute_blocks=True)
    max_num_blocks = triton.cdiv(seq_len, block_size)
    num_splits = triton.cdiv(max_num_blocks, num_blocks_per_split)
    acc = torch.empty((batch_size * n_heads, num_splits, head_dim), dtype=torch.float32, device=device)
    local_max = torch.empty((batch_size * n_heads, num_splits), dtype=torch.float32, device=device)
    local_expsum = torch.empty_like(local_max)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        return triton_kernel.single_query_paged_kv_attention_v2(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            max_num_blocks,
            num_blocks_per_split,
            acc,
            local_max,
            local_expsum,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_paged_attn.run(print_data=True, return_df=True)
    bench_paged_attn_v2_split_size.run(print_data=True, return_df=True)
