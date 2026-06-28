import itertools

import pandas as pd
import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


def torch_swiglu(x, gate):
    return F.silu(gate) * x


compiled_torch_swiglu = torch.compile(torch_swiglu)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['n_elements'],
        x_vals=[2 ** i for i in range(8, 28)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch.compile'],
        line_names=['Triton', 'Torch', 'Torch Compile'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='ms',
        plot_name=f'swiglu-forward-latency num elements (dtype: {str(dtype)})',
    )
])
def bench_swiglu_n_elements(n_elements, provider, device=torch.device('cuda')):
    dtype = torch.bfloat16
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


def perf_kernel_config():
    block_size_params = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    num_warps_params = [1, 2, 4, 8]
    num_stages = 3
    num_ctas = 1

    n_elements = 2 ** 28
    dtype = torch.bfloat16
    x = torch.randn((n_elements,), device='cuda', dtype=dtype) * 0.5 - 2.3
    gate = torch.randn((n_elements,), device='cuda', dtype=dtype)
    y = torch.empty_like(x)

    data = {
        'shape.numel': [],
        'dtype': [],
        'config.block_size': [],
        'config.num_warps': [],
        'config.num_stages': [],
        'config.num_ctas': [],
        'launch.grid_x': [],
        'launch.grid_y': [],
        'launch.grid_z': [],
        'tile.n_regs': [],
        'tile.n_spills': [],
        'tile.n_max_threads': [],
        'tile.shared_mem_bytes': [],
        'perf.latency_ms.p50': [],
        'perf.latency_ms.p20': [],
        'perf.latency_ms.p80': [],
    }

    for block_size, num_warps in itertools.product(block_size_params, num_warps_params):
        grid = (triton.cdiv(y.numel(), block_size), )

        kernel = triton_kernel.swiglu_fwd_fused_kernel.warmup(
            x,
            gate,
            y,
            y.numel(),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            grid=grid,
        )
        kernel._init_handles()

        shared_mem_bytes = kerne.metadata.shared

        p50, p20, p80 = triton.testing.do_bench(
            lambda: triton_kernel.swiglu_fwd_fused_kernel[grid](
                x,
                gate,
                y,
                y.numel(),
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
                num_ctas=num_ctas,
            ),
            rep=500,
            quantiles=[0.5, 0.2, 0.8]
        )

        data['shape.numel'].append(n_elements)
        data['dtype'].append(str(dtype).replace('torch.', ''))
        data['config.block_size'].append(block_size)
        data['config.num_warps'].append(num_warps)
        data['config.num_stages'].append(num_stages)
        data['config.num_ctas'].append(num_ctas)
        data['launch.grid_x'].append(grid[0])
        data['launch.grid_y'].append(1)
        data['launch.grid_z'].append(1)
        data['tile.n_regs'].append(kernel.n_regs)
        data['tile.n_spills'].append(kernel.n_spills)
        data['tile.n_max_threads'].append(kernel.n_max_threads)
        data['tile.shared_mem_bytes'].append(kernel.metadata.shared)
        data['perf.latency_ms.p50'].append(p50)
        data['perf.latency_ms.p20'].append(p20)
        data['perf.latency_ms.p80'].append(p80)

    df = pd.DataFrame(data)
    print(df)
    return df


if __name__ == '__main__':
    # bench_swiglu_n_elements.run(print_data=True, return_df=True)
    perf_kernel_config()
