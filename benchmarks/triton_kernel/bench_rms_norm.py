import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


dtypes = [torch.float16, torch.bfloat16, torch.float32]

benchmarks_hidden_dim = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(16, 32)],
        line_arg='provider',
        line_vals=['triton_1', 'triton_2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('red', '-')],
        ylabel='ms',
        plot_name=f'rms-norm-forward-latency hidden dim (dtype: {str(dtype)})',
        args={'M': 4096, 'dtype': dtype},
    )
    for dtype in dtypes
]

benchmarks_sequence_length = [
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[512 * i for i in range(16, 32)],
        line_arg='provider',
        line_vals=['triton_1', 'triton_2', 'torch', 'torch.compile'],
        line_names=['Triton v1', 'Triton v2', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('red', '-')],
        ylabel='ms',
        plot_name=f'rms-norm-forward-latency sequence length (dtype: {str(dtype)})',
        args={'N': 4096, 'dtype': dtype},
    )
    for dtype in dtypes
]


def torch_rms_norm(x, w, eps):
    return F.rms_norm(x, w.shape, weight=w, eps=eps)


compiled_torch_rms_norm = torch.compile(torch_rms_norm)


@triton.testing.perf_report(benchmarks_hidden_dim + benchmarks_sequence_length)
def bench_rms_norm(M, N, dtype, provider, eps=1e-5, device=torch.device('cuda')):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton_1":
            return triton_kernel.rms_norm_fwd(x, weight, eps)

        if provider == "triton_2":
            return triton_kernel.rms_norm_fwd_v2(x, weight, eps)

        if provider == "torch":
            return torch_rms_norm(x, weight, eps)

        if provider == "torch.compile":
            return compiled_torch_rms_norm(x, weight, eps)

    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_rms_norm.run(print_data=True, return_df=True)
