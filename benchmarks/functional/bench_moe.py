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


compiled_moe = torch.compile(functional.moe)
compiled_torch_dense_moe = torch.compile(torch_dense_moe)


PROVIDERS = ['functional', 'functional.compile', 'torch-dense', 'torch-dense.compile']
PROVIDER_NAMES = ['Functional MoE', 'Functional MoE Compile', 'Torch Dense', 'Torch Dense Compile']
PROVIDER_STYLES = [('blue', '-'), ('red', '-'), ('gray', '-'), ('orange', '-')]


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

    def target_fn():
        if provider == 'functional':
            return functional.moe(x, w_router, w_expert_0, w_expert_1, top_k)

        if provider == 'functional.compile':
            return compiled_moe(x, w_router, w_expert_0, w_expert_1, top_k)

        if provider == 'torch-dense':
            return torch_dense_moe(x, w_router, w_expert_0, w_expert_1, top_k)

        if provider == 'torch-dense.compile':
            return compiled_torch_dense_moe(x, w_router, w_expert_0, w_expert_1, top_k)

        raise ValueError(f'unknown provider: {provider}')

    ms, min_ms, max_ms = triton.testing.do_bench(target_fn, quantiles=quantiles, rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    bench_moe.run(print_data=True, return_df=True)
