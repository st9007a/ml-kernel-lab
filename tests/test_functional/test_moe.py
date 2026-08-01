import pytest
import torch
import torch.nn.functional as F

from ml_kernel_lab import functional


def torch_moe(x, w_router, w_expert_0, w_expert_1, top_k):
    router_logits = x @ w_router
    topk_logits, topk_inds = router_logits.topk(k=top_k, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1)

    hidden = torch.einsum('btd,edf->btef', x, w_expert_0)
    expert_outputs = torch.einsum('btef,efd->bted', F.gelu(hidden), w_expert_1)

    output_indices = topk_inds[..., None].expand(-1, -1, -1, x.shape[-1])
    selected_outputs = torch.gather(expert_outputs, dim=2, index=output_indices)
    return torch.sum(selected_outputs * topk_weights[..., None], dim=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('shape', 'd_ff', 'n_experts', 'top_k', 'tied_router'),
    [
        ((1, 7, 64), 96, 4, 1, False),
        ((2, 5, 72), 130, 5, 2, False),
        ((2, 9, 128), 64, 4, 2, True),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_moe_matches_torch(shape, d_ff, n_experts, top_k, tied_router, dtype):
    _, _, d_model = shape
    x = torch.randn(shape, dtype=dtype, device='cuda')
    w_router = torch.randn((d_model, n_experts), dtype=dtype, device='cuda') / d_model**0.5
    w_expert_0 = torch.randn((n_experts, d_model, d_ff), dtype=dtype, device='cuda') / d_model**0.5
    w_expert_1 = torch.randn((n_experts, d_ff, d_model), dtype=dtype, device='cuda') / d_ff**0.5

    if tied_router:
        w_router.zero_()

    actual = functional.moe(x, w_router, w_expert_0, w_expert_1, top_k)
    expected = torch_moe(x, w_router, w_expert_0, w_expert_1, top_k)

    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
