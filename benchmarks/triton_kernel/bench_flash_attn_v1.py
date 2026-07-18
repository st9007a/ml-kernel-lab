import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


def torch_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def torch_causal_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def torch_attention_unfused(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def torch_causal_attention_unfused(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = q.shape[-2]
    causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(causal_mask, -float('inf'))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


compiled_torch_attention_unfused = torch.compile(torch_attention_unfused)
compiled_torch_causal_attention_unfused = torch.compile(torch_causal_attention_unfused)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024],
        line_arg='provider',
        line_vals=['triton', 'torch_unfused', 'torch_unfused.compile', 'torch_sdpa'],
        line_names=['Triton', 'Torch Unfused', 'Torch Unfused Compile', 'Torch SDPA'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('purple', '-')],
        ylabel='ms',
        plot_name='flash-attn-v1-forward-latency-seq-len',
        args={
            'batch_size': 1,
            'n_heads': 8,
            'head_dim': 128,
        },
    ),
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024],
        line_arg='provider',
        line_vals=['triton', 'torch_unfused', 'torch_unfused.compile', 'torch_sdpa'],
        line_names=['Triton', 'Torch Unfused', 'Torch Unfused Compile', 'Torch SDPA'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('purple', '-')],
        ylabel='ms',
        plot_name='flash-attn-v1-causal-forward-latency-seq-len',
        args={
            'batch_size': 1,
            'n_heads': 8,
            'head_dim': 128,
            'is_causal': True,
        },
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8],
        line_arg='provider',
        line_vals=['triton', 'torch_unfused', 'torch_unfused.compile', 'torch_sdpa'],
        line_names=['Triton', 'Torch Unfused', 'Torch Unfused Compile', 'Torch SDPA'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('purple', '-')],
        ylabel='ms',
        plot_name='flash-attn-v1-forward-latency-batch-size',
        args={
            'seq_len': 512,
            'n_heads': 8,
            'head_dim': 128,
        },
    ),
    triton.testing.Benchmark(
        x_names=['head_dim'],
        x_vals=[64, 128],
        line_arg='provider',
        line_vals=['triton', 'torch_unfused', 'torch_unfused.compile', 'torch_sdpa'],
        line_names=['Triton', 'Torch Unfused', 'Torch Unfused Compile', 'Torch SDPA'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('purple', '-')],
        ylabel='ms',
        plot_name='flash-attn-v1-forward-latency-head-dim',
        args={
            'batch_size': 1,
            'n_heads': 8,
            'seq_len': 512,
        },
    ),
])
def bench_flash_attn_v1(
    batch_size,
    n_heads,
    seq_len,
    head_dim,
    provider,
    is_causal=False,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    q = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.flash_attention_v1_fwd(q, k, v, is_causal=is_causal)

        if provider == 'torch_unfused':
            if is_causal:
                return torch_causal_attention_unfused(q, k, v)
            return torch_attention_unfused(q, k, v)

        if provider == 'torch_unfused.compile':
            if is_causal:
                return compiled_torch_causal_attention_unfused(q, k, v)
            return compiled_torch_attention_unfused(q, k, v)

        if provider == 'torch_sdpa':
            if is_causal:
                return torch_causal_attention(q, k, v)
            return torch_attention(q, k, v)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_flash_attn_v1.run(print_data=True, return_df=True)
