import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


def torch_swiglu(x, gate):
    return F.silu(gate) * x


compiled_torch_swiglu = torch.compile(torch_swiglu)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['intermediate_size'],
        x_vals=[8192, 11008, 14336, 18944, 28672, 57344],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='swiglu-forward-latency-intermediate-size',
        args={'batch_size': 1},
    ),
    triton.testing.Benchmark(
        x_names=['batch_size'],
        x_vals=[1, 2, 4, 8, 16],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name='swiglu-forward-latency-batch-size',
        args={'intermediate_size': 28672},
    ),
])
def bench_swiglu(
    batch_size,
    intermediate_size,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    n_elements = batch_size * intermediate_size
    x = -2.3 + 0.5 * torch.randn((n_elements,), dtype=dtype, device=device)
    gate = torch.randn((n_elements,), dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.swiglu_fwd(x, gate)

        if provider == 'torch':
            return torch_swiglu(x, gate)

        if provider == 'torch.compile':
            return compiled_torch_swiglu(x, gate)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_swiglu.run(print_data=True, return_df=True)
