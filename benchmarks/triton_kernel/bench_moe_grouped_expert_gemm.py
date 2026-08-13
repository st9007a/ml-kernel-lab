import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


PROVIDERS = ['triton', 'triton-autotuned', 'triton-v2', 'triton-v2-autotuned', 'triton-v3', 'torch-grouped-mm']
PROVIDER_NAMES = [
    'Triton v1 Fixed',
    'Triton v1 Autotuned',
    'Triton v2',
    'Triton v2 Autotuned',
    'Triton v3 Persistent',
    'Torch Grouped MM',
]
PROVIDER_STYLES = [('blue', '-'), ('red', '-'), ('cyan', '-'), ('purple', '-'), ('black', '-'), ('green', '-')]
PRODUCTION_PROVIDERS = [
    'triton',
    'triton-autotuned',
    'triton-v2',
    'triton-v2-autotuned',
    'triton-v3',
    'torch-grouped-mm',
    'torch-grouped-mm.compile',
    'torch-grouped-mm.max-autotune',
]
PRODUCTION_PROVIDER_NAMES = [
    'Triton v1 Fixed',
    'Triton v1 Autotuned',
    'Triton v2',
    'Triton v2 Autotuned',
    'Triton v3 Persistent',
    'Torch Grouped MM',
    'Torch Grouped MM Compile',
    'Torch Grouped MM Max Autotune',
]
PRODUCTION_PROVIDER_STYLES = [
    ('blue', '-'),
    ('red', '-'),
    ('cyan', '-'),
    ('magenta', '-'),
    ('black', '-'),
    ('green', '-'),
    ('purple', '-'),
    ('orange', '-'),
]
GROUP_SIZE_M_VALUES = [1, 2, 4, 8, 16, 32]
GROUP_SIZE_M_NAMES = [f'GROUP_SIZE_M={group_size_m}' for group_size_m in GROUP_SIZE_M_VALUES]
GROUP_SIZE_M_STYLES = [
    ('blue', '-'),
    ('red', '-'),
    ('green', '-'),
    ('orange', '-'),
    ('purple', '-'),
    ('cyan', '-'),
]


def grouped_mm_weight_layout(weight):
    return weight.transpose(-2, -1).contiguous().transpose(-2, -1)


def torch_grouped_mm_forward(x_grouped, weight, grouped_mm_offsets):
    return F.grouped_mm(x_grouped, weight, offs=grouped_mm_offsets)


def make_balanced_expert_offsets(num_assignments, n_experts, device):
    assignments_per_expert, remainder = divmod(num_assignments, n_experts)
    counts = [assignments_per_expert + (expert_id < remainder) for expert_id in range(n_experts)]

    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    return torch.tensor(offsets, dtype=torch.int32, device=device), max(counts)


def make_imbalanced_expert_offsets(num_assignments, n_experts, hot_expert_share, device):
    alignment = 16
    if num_assignments % alignment != 0:
        raise ValueError('num_assignments must be divisible by 16')

    num_chunks = num_assignments // alignment
    hot_chunks = round(num_chunks * hot_expert_share)
    remaining_chunks = num_chunks - hot_chunks
    chunks_per_expert, remainder = divmod(remaining_chunks, n_experts - 1)

    counts = [hot_chunks * alignment]
    counts.extend(
        (chunks_per_expert + (expert_id < remainder)) * alignment
        for expert_id in range(n_experts - 1)
    )
    if min(counts) == 0:
        raise ValueError('hot_expert_share leaves at least one expert empty')

    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    return torch.tensor(offsets, dtype=torch.int32, device=device), max(counts)


def launch_moe_grouped_expert_gemm_v2(
    x_grouped,
    weight,
    expert_offsets,
    out,
    expert_capacity,
    group_size_m,
):
    m, k = x_grouped.shape
    n_experts, _, n = weight.shape
    block_m = 64
    block_n = 128
    block_k = 32
    max_m_tiles = triton.cdiv(expert_capacity, block_m)
    grid = (max_m_tiles * triton.cdiv(n, block_n), n_experts)

    triton_kernel.moe_grouped_expert_gemm_fwd_v2_fused_kernel[grid](
        x_grouped,
        weight,
        expert_offsets,
        out,
        x_grouped.stride(0),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        m,
        n,
        k,
        n_experts,
        expert_capacity,
        group_size_m,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['num_assignments'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-num-assignments',
        args={
            'n_experts': 8,
            'd_model': 256,
            'd_ff': 1024,
        },
    ),
    triton.testing.Benchmark(
        x_names=['n_experts'],
        x_vals=[4, 8, 16, 32],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-num-experts',
        args={
            'num_assignments': 4096,
            'd_model': 256,
            'd_ff': 1024,
        },
    ),
    triton.testing.Benchmark(
        x_names=['d_model', 'd_ff'],
        x_vals=[(128, 512), (256, 1024), (512, 2048)],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-model-size',
        args={
            'num_assignments': 1024,
            'n_experts': 8,
        },
    ),
])
def bench_moe_grouped_expert_gemm(
    num_assignments,
    n_experts,
    d_model,
    d_ff,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    grouped_mm = getattr(F, 'grouped_mm', None)
    if provider == 'torch-grouped-mm' and grouped_mm is None:
        raise RuntimeError(
            f'torch.nn.functional.grouped_mm is unavailable in PyTorch {torch.__version__}; '
            'install a PyTorch build that provides the public grouped_mm API'
        )

    expert_offsets, expert_capacity = make_balanced_expert_offsets(num_assignments, n_experts, device)

    # grouped_mm requires offs[-1] to be smaller than the input length. Triton
    # uses an exact-length view of the same storage so its M autotune key is exact.
    generator = torch.Generator(device=device).manual_seed(0)
    x_grouped = torch.randn(
        (num_assignments + 1, d_model),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    x_triton = x_grouped[:num_assignments]
    w = torch.randn(
        (n_experts, d_model, d_ff),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    w_grouped_mm = grouped_mm_weight_layout(w)
    grouped_mm_offsets = expert_offsets[1:]
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v3':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v3(
                x_triton,
                w,
                expert_offsets,
                num_sms,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w_grouped_mm, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

    if provider in ('triton-autotuned', 'triton-v2-autotuned'):
        target_fn()
        torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['hot_expert_share'],
        x_vals=[0.125, 0.25, 0.5, 0.75, 0.875],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-routing-imbalance',
        args={
            'num_assignments': 4096,
            'n_experts': 8,
            'd_model': 256,
            'd_ff': 1024,
        },
    )
])
def bench_moe_grouped_expert_gemm_imbalance(
    num_assignments,
    n_experts,
    d_model,
    d_ff,
    hot_expert_share,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    grouped_mm = getattr(F, 'grouped_mm', None)
    if provider == 'torch-grouped-mm' and grouped_mm is None:
        raise RuntimeError(
            f'torch.nn.functional.grouped_mm is unavailable in PyTorch {torch.__version__}; '
            'install a PyTorch build that provides the public grouped_mm API'
        )

    expert_offsets, expert_capacity = make_imbalanced_expert_offsets(
        num_assignments,
        n_experts,
        hot_expert_share,
        device,
    )

    generator = torch.Generator(device=device).manual_seed(0)
    x_grouped = torch.randn(
        (num_assignments + 1, d_model),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    x_triton = x_grouped[:num_assignments]
    w = torch.randn(
        (n_experts, d_model, d_ff),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    w_grouped_mm = grouped_mm_weight_layout(w)
    grouped_mm_offsets = expert_offsets[1:]
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v3':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v3(
                x_triton,
                w,
                expert_offsets,
                num_sms,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w_grouped_mm, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

    if provider in ('triton-autotuned', 'triton-v2-autotuned'):
        target_fn()
        torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['expert_capacity'],
        x_vals=[512, 1024, 1536, 2048],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-expert-capacity',
        args={
            'num_tokens': 2048,
            'top_k': 2,
            'n_experts': 8,
            'd_model': 256,
            'd_ff': 1024,
        },
    )
])
def bench_moe_grouped_expert_gemm_capacity(
    num_tokens,
    top_k,
    n_experts,
    d_model,
    d_ff,
    expert_capacity,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    grouped_mm = getattr(F, 'grouped_mm', None)
    if provider == 'torch-grouped-mm' and grouped_mm is None:
        raise RuntimeError(
            f'torch.nn.functional.grouped_mm is unavailable in PyTorch {torch.__version__}; '
            'install a PyTorch build that provides the public grouped_mm API'
        )

    num_assignments = num_tokens * top_k
    expert_offsets, actual_expert_capacity = make_balanced_expert_offsets(
        num_assignments,
        n_experts,
        device,
    )
    if expert_capacity < actual_expert_capacity:
        raise ValueError('expert_capacity must cover the largest expert assignment count')
    if expert_capacity > num_tokens:
        raise ValueError('expert_capacity cannot exceed the dropless top-k upper bound')

    generator = torch.Generator(device=device).manual_seed(0)
    x_grouped = torch.randn(
        (num_assignments + 1, d_model),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    x_triton = x_grouped[:num_assignments]
    w = torch.randn(
        (n_experts, d_model, d_ff),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    w_grouped_mm = grouped_mm_weight_layout(w)
    grouped_mm_offsets = expert_offsets[1:]
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v2-autotuned':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2_autotuned(
                x_triton,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'triton-v3':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v3(
                x_triton,
                w,
                expert_offsets,
                num_sms,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w_grouped_mm, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

    if provider in ('triton-autotuned', 'triton-v2-autotuned'):
        target_fn()
        torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['d_model', 'd_ff'],
        x_vals=[(1024, 4096), (2048, 8192), (4096, 14336)],
        line_arg='provider',
        line_vals=PRODUCTION_PROVIDERS,
        line_names=PRODUCTION_PROVIDER_NAMES,
        styles=PRODUCTION_PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-production-model-size',
        args={
            'num_assignments': 4096,
            'n_experts': 8,
        },
    ),
    triton.testing.Benchmark(
        x_names=['num_assignments'],
        x_vals=[4096, 8192, 16384],
        line_arg='provider',
        line_vals=PRODUCTION_PROVIDERS,
        line_names=PRODUCTION_PROVIDER_NAMES,
        styles=PRODUCTION_PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-forward-latency-production-num-assignments',
        args={
            'n_experts': 8,
            'd_model': 4096,
            'd_ff': 14336,
        },
    ),
])
def bench_moe_grouped_expert_gemm_production(
    num_assignments,
    n_experts,
    d_model,
    d_ff,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    grouped_mm = getattr(F, 'grouped_mm', None)
    if not provider.startswith('triton') and grouped_mm is None:
        raise RuntimeError(
            f'torch.nn.functional.grouped_mm is unavailable in PyTorch {torch.__version__}; '
            'install a PyTorch build that provides the public grouped_mm API'
        )

    expert_offsets, expert_capacity = make_balanced_expert_offsets(
        num_assignments,
        n_experts,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    x_grouped = torch.randn(
        (num_assignments + 1, d_model),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    x_triton = x_grouped[:num_assignments]
    weight = torch.randn(
        (n_experts, d_model, d_ff),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    grouped_mm_offsets = expert_offsets[1:]
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        def target_fn():
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_triton,
                weight,
                expert_offsets,
                expert_capacity,
            )
    elif provider == 'triton-autotuned':
        def target_fn():
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1_autotuned(
                x_triton,
                weight,
                expert_offsets,
                expert_capacity,
            )
    elif provider == 'triton-v2':
        def target_fn():
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2(
                x_triton,
                weight,
                expert_offsets,
                expert_capacity,
            )
    elif provider == 'triton-v2-autotuned':
        def target_fn():
            return triton_kernel.moe_grouped_expert_gemm_fwd_v2_autotuned(
                x_triton,
                weight,
                expert_offsets,
                expert_capacity,
            )
    elif provider == 'triton-v3':
        def target_fn():
            return triton_kernel.moe_grouped_expert_gemm_fwd_v3(
                x_triton,
                weight,
                expert_offsets,
                num_sms,
            )
    else:
        weight = grouped_mm_weight_layout(weight)
        if provider == 'torch-grouped-mm':
            target = torch_grouped_mm_forward
        elif provider == 'torch-grouped-mm.compile':
            torch._dynamo.reset()
            target = torch.compile(
                torch_grouped_mm_forward,
                options={'triton.cudagraphs': False},
            )
        elif provider == 'torch-grouped-mm.max-autotune':
            torch._dynamo.reset()
            target = torch.compile(
                torch_grouped_mm_forward,
                mode='max-autotune-no-cudagraphs',
            )
        else:
            raise ValueError(f'unknown provider: {provider}')

        def target_fn():
            return target(x_grouped, weight, grouped_mm_offsets)

        if provider.endswith(('.compile', '.max-autotune')):
            target_fn()
            torch.cuda.synchronize()

    if provider in ('triton-autotuned', 'triton-v2-autotuned'):
        target_fn()
        torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['num_assignments'],
        x_vals=[4096, 8192, 16384],
        line_arg='group_size_m',
        line_vals=GROUP_SIZE_M_VALUES,
        line_names=GROUP_SIZE_M_NAMES,
        styles=GROUP_SIZE_M_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-v2-forward-latency-group-size-production',
        args={
            'n_experts': 8,
            'd_model': 4096,
            'd_ff': 14336,
            'routing': 'balanced',
        },
    ),
    triton.testing.Benchmark(
        x_names=['hot_expert_share'],
        x_vals=[0.125, 0.25, 0.5, 0.75, 0.875],
        line_arg='group_size_m',
        line_vals=GROUP_SIZE_M_VALUES,
        line_names=GROUP_SIZE_M_NAMES,
        styles=GROUP_SIZE_M_STYLES,
        ylabel='ms',
        plot_name='moe-grouped-expert-gemm-v2-forward-latency-group-size-imbalance',
        args={
            'num_assignments': 4096,
            'n_experts': 8,
            'd_model': 256,
            'd_ff': 1024,
            'routing': 'imbalanced',
        },
    ),
])
def bench_moe_grouped_expert_gemm_v2_group_size(
    num_assignments,
    n_experts,
    d_model,
    d_ff,
    routing,
    group_size_m,
    hot_expert_share=None,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    if routing == 'balanced':
        expert_offsets, expert_capacity = make_balanced_expert_offsets(
            num_assignments,
            n_experts,
            device,
        )
    elif routing == 'imbalanced':
        expert_offsets, expert_capacity = make_imbalanced_expert_offsets(
            num_assignments,
            n_experts,
            hot_expert_share,
            device,
        )
    else:
        raise ValueError(f'unknown routing: {routing}')

    generator = torch.Generator(device=device).manual_seed(0)
    x_grouped = torch.randn(
        (num_assignments, d_model),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        (n_experts, d_model, d_ff),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    out = torch.empty(
        (num_assignments, d_ff),
        dtype=dtype,
        device=device,
    )
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        return launch_moe_grouped_expert_gemm_v2(
            x_grouped,
            weight,
            expert_offsets,
            out,
            expert_capacity,
            group_size_m,
        )

    # Compile the specialization before measuring kernel launch latency.
    target_fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    if not hasattr(F, 'grouped_mm'):
        raise RuntimeError(
            f'torch.nn.functional.grouped_mm is unavailable in PyTorch {torch.__version__}; '
            'install a PyTorch build that provides the public grouped_mm API'
        )

    bench_moe_grouped_expert_gemm.run(print_data=True, return_df=True)
    bench_moe_grouped_expert_gemm_imbalance.run(print_data=True, return_df=True)
    bench_moe_grouped_expert_gemm_capacity.run(print_data=True, return_df=True)
    bench_moe_grouped_expert_gemm_production.run(print_data=True, return_df=True)
    bench_moe_grouped_expert_gemm_v2_group_size.run(print_data=True, return_df=True)
