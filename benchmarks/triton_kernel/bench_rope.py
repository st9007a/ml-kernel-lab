import torch
import triton

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


def torch_rope_one(x, cos, sin):
    batch_size, n_heads, seq_len, head_dim = x.shape
    cos, sin = expand_cos_sin(cos, sin, batch_size, n_heads, seq_len, head_dim)
    return x * cos + rotate_half(x) * sin


def torch_rope(q, k, cos, sin):
    return torch_rope_one(q, cos, sin), torch_rope_one(k, cos, sin)


compiled_torch_rope = torch.compile(torch_rope)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='rope-forward-latency-seq-len',
        args={
            'batch_size': 1,
            'n_q_head': 32,
            'n_k_head': 8,
            'head_dim': 128,
        },
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='rope-forward-latency-batch-size',
        args={
            'seq_len': 1,
            'n_q_head': 32,
            'n_k_head': 8,
            'head_dim': 128,
        },
    ),
])
def bench_rope(
    batch_size,
    seq_len,
    n_q_head,
    n_k_head,
    head_dim,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    q = torch.randn((batch_size, seq_len, n_q_head, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, seq_len, n_k_head, head_dim), dtype=dtype, device=device)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    angles = torch.randn((1, seq_len, head_dim // 2), dtype=torch.float32, device=device)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)

    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.rope_fwd(q, k, cos, sin)

        if provider == 'torch':
            return torch_rope(q, k, cos, sin)

        if provider == 'torch.compile':
            return compiled_torch_rope(q, k, cos, sin)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_rope.run(print_data=True, return_df=True)
