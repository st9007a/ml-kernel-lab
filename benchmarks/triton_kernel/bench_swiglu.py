import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


dtypes = [torch.float16, torch.bfloat16, torch.float32]

benchmarks_num_elements = [
    triton.testing.Benchmark(
        x_names=['n_elements'],
        x_vals=[2 ** i for i in range(8, 28)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name=f'swiglu-forward-latency num elements (dtype: {str(dtype)})',
        args={'dtype': dtype},
    )
    for dtype in dtypes
]

benchmarks_triton_config = [
    triton.testing.Benchmark(
        x_names=['block size, num warps'],
        x_vals=list(itertools.product([512, 1024, 2048, 4096, 8192], [1, 2, 4, 8])),
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('blue', '-')],
        ylabel='ms',
        plot_name=f'swiglu-block-size-num-warps (dtype: {str(dtype)})',
        args={'dtype': dtype},
    )
    for dtype in dtypes
]


def torch_swiglu(x, gate):
    return F.silu(gate) * x


compiled_torch_swiglu = torch.compile(torch_swiglu)


@triton.testing.perf_report(benchmarks_num_elements)
def bench_swiglu_n_elements(n_elements, dtype, provider, device=torch.device('cuda')):
    x = -2.3 + 0.5 * torch.randn((n_elements,), dtype=dtype, device=device)
    gate = torch.randn((n_elements,), dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == 'triton':
            return triton_kernel.swiglu_fwd(x, gate)

        if provider == 'torch':
            return torch_swiglu(x, gate)

        if provider == 'torch.compile':
            return compiled_torch_swiglu(x, gate)

    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return ms, max_ms, min_ms


@triton.testing.perf_report(benchmarks_triton_config)
def bench_swiglu_triton_config(config, dtype, provider, device=torch.device('cuda')):
    n_elements = 8192
    blick_size, num_warps = config
    grid = (triton.cdiv(n_elements, block_size), )
    x = torch.randn((n_elements,), dtype=dtype, device=device) * 0.5 - 2.3
    gate = torch.randn((n_elements,), dtype=dtype, device=device)
    y = torch.empty_like(x)
    quantiles = [0.5, 0.2, 0.8]


    def target_fn():
        triton_kernel.swiglu_fwd_fused_kernel[grid](
            x,
            gate,
            y,
            n_elements,
            block_size,
            num_warps=num_warps,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, min_ms, max_ms


if __name__ == '__main__':
    bench_swiglu_n_elements.run(print_data=True, return_df=True)
    bench_swiglu_triton_config.run(print_data=True, return_df=True)
