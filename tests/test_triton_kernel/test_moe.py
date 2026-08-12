import pytest
import torch

from ml_kernel_lab.kernel import triton_kernel


def torch_grouped_expert_gemm(x_grouped, w, expert_offsets):
    out = torch.empty((x_grouped.shape[0], w.shape[2]), dtype=x_grouped.dtype, device=x_grouped.device)

    for expert_id in range(w.shape[0]):
        start = int(expert_offsets[expert_id].item())
        end = int(expert_offsets[expert_id + 1].item())
        if start == end:
            continue
        out[start:end] = x_grouped[start:end] @ w[expert_id]

    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('expert_counts', 'k', 'n'),
    [
        ([16, 16, 16], 64, 64),
        ([1, 0, 17, 5], 64, 96),
        ([31, 8, 93, 0, 2], 72, 130),
        ([3, 33, 65], 128, 257),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_grouped_expert_gemm_v1_matches_torch(expert_counts, k, n, dtype):
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    n_experts = len(expert_counts)
    expert_capacity = max(expert_counts)

    x_grouped = torch.randn((total_assignments, k), dtype=dtype, device='cuda')
    w = torch.randn((n_experts, k, n), dtype=dtype, device='cuda')
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v1(x_grouped, w, expert_offsets, expert_capacity)
    expected = torch_grouped_expert_gemm(x_grouped, w, expert_offsets)

    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_grouped_expert_gemm_v1_accepts_non_contiguous_inputs():
    dtype = torch.float16
    expert_counts = [7, 19, 4]
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    k = 64
    n = 96
    n_experts = len(expert_counts)

    x_base = torch.randn((k, total_assignments), dtype=dtype, device='cuda')
    x_grouped = x_base.t()
    w_base = torch.randn((n_experts, n, k), dtype=dtype, device='cuda')
    w = w_base.transpose(1, 2)
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v1(x_grouped, w, expert_offsets, max(expert_counts))
    expected = torch_grouped_expert_gemm(x_grouped.contiguous(), w.contiguous(), expert_offsets)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_grouped_expert_gemm_v1_autotuned_matches_torch_with_overlaunch():
    dtype = torch.bfloat16
    expert_counts = [31, 8, 93, 0, 2]
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    n_experts = len(expert_counts)
    k = 72
    n = 130
    # Deliberately larger than the actual maximum to exercise the config-dependent grid.
    expert_capacity = 128

    x_grouped = torch.randn((total_assignments, k), dtype=dtype, device='cuda')
    w = torch.randn((n_experts, k, n), dtype=dtype, device='cuda')
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v1_autotuned(
        x_grouped,
        w,
        expert_offsets,
        expert_capacity,
    )
    expected = torch_grouped_expert_gemm(x_grouped, w, expert_offsets)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('expert_counts', 'k', 'n', 'expert_capacity'),
    [
        ([1, 0, 17, 5], 64, 257, 64),
        ([577, 33, 0, 129], 72, 257, 704),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_grouped_expert_gemm_v2_matches_torch(
    expert_counts,
    k,
    n,
    expert_capacity,
    dtype,
):
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    n_experts = len(expert_counts)

    x_grouped = torch.randn((total_assignments, k), dtype=dtype, device='cuda')
    w = torch.randn((n_experts, k, n), dtype=dtype, device='cuda')
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v2(
        x_grouped,
        w,
        expert_offsets,
        expert_capacity,
    )
    expected = torch_grouped_expert_gemm(x_grouped, w, expert_offsets)

    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('expert_counts', 'k', 'n', 'expert_capacity'),
    [
        ([1, 0, 17, 5], 72, 257, 64),
        ([129, 65, 0, 257], 128, 130, 384),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_grouped_expert_gemm_v2_autotuned_matches_torch(
    expert_counts,
    k,
    n,
    expert_capacity,
    dtype,
):
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    n_experts = len(expert_counts)
    generator = torch.Generator(device='cuda').manual_seed(0)

    x_grouped = torch.randint(
        -1,
        2,
        (total_assignments, k),
        device='cuda',
        generator=generator,
    ).to(dtype)
    w = torch.randint(
        -1,
        2,
        (n_experts, k, n),
        device='cuda',
        generator=generator,
    ).to(dtype)
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v2_autotuned(
        x_grouped,
        w,
        expert_offsets,
        expert_capacity,
    )
    expected = torch_grouped_expert_gemm(x_grouped, w, expert_offsets)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    ('expert_counts', 'k', 'n', 'num_sms'),
    [
        # More programs than tiles exercises persistent-grid overlaunch.
        ([1, 0, 17, 5], 72, 257, 128),
        # Two programs must repeatedly cross expert and grouped-M boundaries.
        ([577, 33, 0, 129], 72, 257, 2),
        # Eight experts require space for nine offsets, padded to sixteen.
        ([0, 1, 0, 65, 3, 0, 129, 7], 128, 130, 3),
    ],
)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_grouped_expert_gemm_v3_matches_torch_with_persistent_scheduling(
    expert_counts,
    k,
    n,
    num_sms,
    dtype,
):
    expert_offsets_values = [0]
    for count in expert_counts:
        expert_offsets_values.append(expert_offsets_values[-1] + count)

    total_assignments = expert_offsets_values[-1]
    n_experts = len(expert_counts)
    generator = torch.Generator(device='cuda').manual_seed(0)

    # Integer-valued inputs isolate scheduler errors from reduction-order noise.
    x_grouped = torch.randint(
        -1,
        2,
        (total_assignments, k),
        device='cuda',
        generator=generator,
    ).to(dtype)
    w = torch.randint(
        -1,
        2,
        (n_experts, k, n),
        device='cuda',
        generator=generator,
    ).to(dtype)
    expert_offsets = torch.tensor(expert_offsets_values, dtype=torch.int32, device='cuda')

    actual = triton_kernel.moe_grouped_expert_gemm_fwd_v3(
        x_grouped,
        w,
        expert_offsets,
        num_sms,
    )
    expected = torch_grouped_expert_gemm(x_grouped, w, expert_offsets)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
