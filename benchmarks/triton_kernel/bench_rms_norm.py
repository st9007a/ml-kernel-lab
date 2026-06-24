import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-forward',
        args={'M': 4096, 'dtype': torch.float16},
    ))
def bench_rms_norm(M, N, dtype, provider, eps=1e-5, device=torch.device('cuda')):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return triton_kernel.rms_norm_fwd(x, weight, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.rms_norm(x, w_shape, weight=weight, eps=eps)  # noqa: F811, E704

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    bench_rms_norm.run(print_data=True)
