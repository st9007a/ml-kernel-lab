import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab.kernel import triton_kernel


PROVIDERS = ['triton', 'torch-grouped-mm']
PROVIDER_NAMES = ['Triton', 'Torch Grouped MM']
PROVIDER_STYLES = [('blue', '-'), ('green', '-')]


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

    # grouped_mm requires offs[-1] to be smaller than the input length. Both
    # providers receive the same padded storage and ignore its final row.
    x_grouped = torch.randn((num_assignments + 1, d_model), dtype=dtype, device=device)
    w = torch.randn((n_experts, d_model, d_ff), dtype=dtype, device=device)
    grouped_mm_offsets = expert_offsets[1:]
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_grouped,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

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

    x_grouped = torch.randn((num_assignments + 1, d_model), dtype=dtype, device=device)
    w = torch.randn((n_experts, d_model, d_ff), dtype=dtype, device=device)
    grouped_mm_offsets = expert_offsets[1:]
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_grouped,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

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

    x_grouped = torch.randn((num_assignments + 1, d_model), dtype=dtype, device=device)
    w = torch.randn((n_experts, d_model, d_ff), dtype=dtype, device=device)
    grouped_mm_offsets = expert_offsets[1:]
    quantiles = [0.5, 0.2, 0.8]

    def target_fn():
        if provider == 'triton':
            return triton_kernel.moe_grouped_expert_gemm_fwd_v1(
                x_grouped,
                w,
                expert_offsets,
                expert_capacity,
            )

        if provider == 'torch-grouped-mm':
            return grouped_mm(x_grouped, w, offs=grouped_mm_offsets)

        raise ValueError(f'unknown provider: {provider}')

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
