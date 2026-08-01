import torch
import torch.nn.functional as F

from ml_kernel_lab.kernel.triton_kernel import moe_grouped_expert_gemm_fwd_v1


def moe(x: torch.Tensor, w_router: torch.Tensor, w_expert_0: torch.Tensor, w_expert_1: torch.Tensor, top_k) -> torch.Tensor:
    """
    x: [B, T, D_model]
    w_router: [D_model, n_experts]
    w_export: [n_experts, D_model, D_ff]

    out: [B, T, D_model]
    """
    n_experts = w_expert_0.shape[0]
    B, T, D = x.shape
    expert_capacity = B * T

    router_logits = x @ w_router
    topk_logits, topk_inds = router_logits.topk(k=top_k, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1)

    x_grouped, expert_offsets, token_ids_sorted, router_weights_sorted = _build_moe_routing(
        x,
        topk_weights,
        topk_inds,
        n_experts,
    )

    hidden_grouped = moe_grouped_expert_gemm_fwd_v1(
        x_grouped,
        w_expert_0,
        expert_offsets,
        expert_capacity,
    )

    hidden_grouped = F.gelu(hidden_grouped)

    logits_grouped = moe_grouped_expert_gemm_fwd_v1(
        hidden_grouped,
        w_expert_1,
        expert_offsets,
        expert_capacity,
    )

    weighted = logits_grouped * router_weights_sorted[:, None]

    output = torch.zeros((B * T, D), dtype=x.dtype, device=x.device)
    output.index_add_(0, token_ids_sorted, weighted)
    return output.reshape(B, T, D)


def _build_moe_routing(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_inds: torch.Tensor,
    n_experts: int,
):
    B, T, D = x.shape
    top_k = topk_inds.shape[-1]

    tokens = x.reshape(B * T, D)
    expert_ids = topk_inds.reshape(-1)
    router_weights = topk_weights.reshape(-1)
    token_ids = torch.arange(B * T, dtype=torch.int64, device=x.device).repeat_interleave(top_k)

    sort_idx = torch.argsort(expert_ids, stable=True)

    token_ids_sorted = token_ids[sort_idx]            # [B * T * K]
    router_weights_sorted = router_weights[sort_idx]  # [B * T * K]

    x_grouped = tokens[token_ids_sorted]

    counts = torch.zeros((n_experts, ), device=x.device, dtype=torch.int32)
    counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32))

    expert_offsets = torch.empty((n_experts + 1,), dtype=torch.int32, device=x.device)
    expert_offsets[0] = 0
    expert_offsets[1:] = counts
    expert_offsets = torch.cumsum(expert_offsets, dim=0)

    return x_grouped, expert_offsets, token_ids_sorted, router_weights_sorted
