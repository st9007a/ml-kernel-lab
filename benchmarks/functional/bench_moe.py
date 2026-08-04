import torch
import torch.nn.functional as F
import triton

from ml_kernel_lab import functional


def torch_dense_moe(x, w_router, w_expert_0, w_expert_1, top_k):
    router_logits = x @ w_router
    topk_logits, topk_inds = router_logits.topk(k=top_k, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1)

    hidden = torch.einsum('btd,edf->btef', x, w_expert_0)
    expert_outputs = torch.einsum('btef,efd->bted', F.gelu(hidden), w_expert_1)

    output_indices = topk_inds[..., None].expand(-1, -1, -1, x.shape[-1])
    selected_outputs = torch.gather(expert_outputs, dim=2, index=output_indices)
    return torch.sum(selected_outputs * topk_weights[..., None], dim=2)


def build_torch_grouped_mm_routing(x, topk_weights, topk_inds, n_experts):
    batch_size, seq_len, d_model = x.shape
    top_k = topk_inds.shape[-1]

    tokens = x.reshape(batch_size * seq_len, d_model)
    expert_ids = topk_inds.reshape(-1)
    router_weights = topk_weights.reshape(-1)
    token_ids = torch.arange(
        batch_size * seq_len,
        dtype=torch.int64,
        device=x.device,
    ).repeat_interleave(top_k)

    sort_idx = torch.argsort(expert_ids, stable=True)
    token_ids_sorted = token_ids[sort_idx]
    router_weights_sorted = router_weights[sort_idx]

    padded_token_ids = torch.cat((token_ids_sorted, token_ids_sorted.new_zeros((1,))))
    x_grouped = tokens[padded_token_ids]

    counts = torch.zeros((n_experts,), device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32))
    expert_offsets = torch.empty((n_experts + 1,), dtype=torch.int32, device=x.device)
    expert_offsets[0] = 0
    expert_offsets[1:] = counts
    expert_offsets = torch.cumsum(expert_offsets, dim=0)

    return x_grouped, expert_offsets, token_ids_sorted, router_weights_sorted


def torch_grouped_mm_moe(x, w_router, w_expert_0, w_expert_1, top_k):
    n_experts = w_expert_0.shape[0]
    batch_size, seq_len, d_model = x.shape

    router_logits = x @ w_router
    topk_logits, topk_inds = router_logits.topk(k=top_k, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1)

    x_grouped, expert_offsets, token_ids_sorted, router_weights_sorted = build_torch_grouped_mm_routing(
        x,
        topk_weights,
        topk_inds,
        n_experts,
    )

    grouped_mm_offsets = expert_offsets[1:]
    hidden_grouped = F.grouped_mm(x_grouped, w_expert_0, offs=grouped_mm_offsets)
    hidden_grouped = F.gelu(hidden_grouped)
    logits_grouped = F.grouped_mm(hidden_grouped, w_expert_1, offs=grouped_mm_offsets)

    num_assignments = token_ids_sorted.shape[0]
    weighted = logits_grouped[:num_assignments] * router_weights_sorted[:, None]
    output = torch.zeros((batch_size * seq_len, d_model), dtype=x.dtype, device=x.device)
    output.index_add_(0, token_ids_sorted, weighted)
    return output.reshape(batch_size, seq_len, d_model)


PROVIDERS = [
    'functional',
    'functional.compile',
    'torch-grouped-mm',
    'torch-grouped-mm.compile',
    'torch-dense',
    'torch-dense.compile',
]
PROVIDER_NAMES = [
    'Functional MoE',
    'Functional MoE Compile',
    'Torch Grouped MM MoE',
    'Torch Grouped MM MoE Compile',
    'Torch Dense',
    'Torch Dense Compile',
]
PROVIDER_STYLES = [
    ('blue', '-'),
    ('red', '-'),
    ('green', '-'),
    ('purple', '-'),
    ('gray', '-'),
    ('orange', '-'),
]


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['num_tokens'],
        x_vals=[128, 256, 512, 1024, 2048],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-forward-latency-num-tokens',
        args={
            'd_model': 256,
            'd_ff': 1024,
            'n_experts': 8,
            'top_k': 2,
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
        plot_name='moe-forward-latency-num-experts',
        args={
            'num_tokens': 512,
            'd_model': 256,
            'd_ff': 1024,
            'top_k': 2,
        },
    ),
    triton.testing.Benchmark(
        x_names=['top_k'],
        x_vals=[1, 2, 4],
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel='ms',
        plot_name='moe-forward-latency-top-k',
        args={
            'num_tokens': 512,
            'd_model': 256,
            'd_ff': 1024,
            'n_experts': 8,
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
        plot_name='moe-forward-latency-model-size',
        args={
            'num_tokens': 512,
            'n_experts': 8,
            'top_k': 2,
        },
    ),
])
def bench_moe(
    num_tokens,
    d_model,
    d_ff,
    n_experts,
    top_k,
    provider,
    device=torch.device('cuda'),
):
    dtype = torch.bfloat16
    x = torch.randn((1, num_tokens, d_model), dtype=dtype, device=device)
    w_router = torch.randn((d_model, n_experts), dtype=dtype, device=device) / d_model**0.5
    w_expert_0 = torch.randn((n_experts, d_model, d_ff), dtype=dtype, device=device) / d_model**0.5
    w_expert_1 = torch.randn((n_experts, d_ff, d_model), dtype=dtype, device=device) / d_ff**0.5
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'functional':
        target = functional.moe
    elif provider == 'functional.compile':
        torch._dynamo.reset()
        target = torch.compile(functional.moe)
    elif provider == 'torch-grouped-mm':
        target = torch_grouped_mm_moe
    elif provider == 'torch-grouped-mm.compile':
        torch._dynamo.reset()
        target = torch.compile(torch_grouped_mm_moe)
    elif provider == 'torch-dense':
        target = torch_dense_moe
    elif provider == 'torch-dense.compile':
        torch._dynamo.reset()
        target = torch.compile(torch_dense_moe)
    else:
        raise ValueError(f'unknown provider: {provider}')

    def target_fn():
        return target(x, w_router, w_expert_0, w_expert_1, top_k)

    if provider.endswith('.compile'):
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

    bench_moe.run(print_data=True, return_df=True)
