import torch
import triton

from ml_kernel_lab.kernel import triton_kernel


def torch_masked_softmax(x, attn_mask):
    return torch.softmax(x.masked_fill(attn_mask != 0, -float('inf')), dim=-1)


compiled_torch_masked_softmax = torch.compile(torch_masked_softmax)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['kv_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['triton_v1', 'triton_v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('purple', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='masked-softmax-forward-latency-decode-kv-len',
        args={
            'batch_size': 1,
            'n_heads': 32,
            'q_len': 1,
            'mask_q_len': 1,
        },
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16],
        line_arg='provider',
        line_vals=['triton_v1', 'triton_v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('purple', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='masked-softmax-forward-latency-batch-decode',
        args={
            'n_heads': 32,
            'q_len': 1,
            'kv_len': 2048,
            'mask_q_len': 1,
        },
    ),
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048],
        line_arg='provider',
        line_vals=['triton_v1', 'triton_v2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('purple', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='masked-softmax-forward-latency-prefill-seq-len',
        args={
            'batch_size': 1,
            'n_heads': 32,
            'mask_q_len': 'full',
        },
    ),
])
def bench_masked_softmax(
    batch_size,
    n_heads,
    provider,
    q_len=None,
    kv_len=None,
    seq_len=None,
    mask_q_len='full',
    device=torch.device('cuda'),
):
    if seq_len is not None:
        q_len = seq_len
        kv_len = seq_len

    dtype = torch.bfloat16
    x = torch.randn((batch_size, n_heads, q_len, kv_len), dtype=dtype, device=device)
    attn_mask_q_len = 1 if mask_q_len == 1 else q_len
    attn_mask = torch.rand((batch_size, 1, attn_mask_q_len, kv_len), device=device) < 0.15
    attn_mask[..., 0] = False
    expanded_attn_mask = attn_mask.expand(batch_size, n_heads, q_len, kv_len).contiguous()
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton_v1':
            return triton_kernel.masked_softmax_fwd(x, attn_mask)

        if provider == 'triton_v2':
            return triton_kernel.masked_softmax_fwd_v2(x, expanded_attn_mask)

        if provider == 'torch':
            return torch_masked_softmax(x, attn_mask)

        if provider == 'torch.compile':
            return compiled_torch_masked_softmax(x, attn_mask)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_masked_softmax.run(print_data=True, return_df=True)
