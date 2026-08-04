import argparse
from dataclasses import dataclass
from typing import Any

import torch
import triton

from ml_kernel_lab.kernel.triton_kernel.moe import moe_grouped_expert_gemm_fwd_v1_fused_kernel


BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 3


@dataclass(frozen=True)
class Case:
    name: str
    distribution: str
    expert_counts: tuple[int, ...]
    d_model: int = 256
    d_ff: int = 1024

    @property
    def num_assignments(self) -> int:
        return sum(self.expert_counts)

    @property
    def n_experts(self) -> int:
        return len(self.expert_counts)

    @property
    def expert_capacity(self) -> int:
        return max(self.expert_counts)


def balanced_counts(num_assignments: int, n_experts: int) -> tuple[int, ...]:
    assignments_per_expert, remainder = divmod(num_assignments, n_experts)
    return tuple(
        assignments_per_expert + (expert_id < remainder)
        for expert_id in range(n_experts)
    )


def hot_expert_counts(
    num_assignments: int,
    n_experts: int,
    hot_expert_share: float,
) -> tuple[int, ...]:
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
    return tuple(counts)


CASES = [
    Case('assignments_128', 'balanced', balanced_counts(128, 8)),
    Case('balanced_4096', 'balanced', balanced_counts(4096, 8)),
    Case('experts_4', 'balanced', balanced_counts(4096, 4)),
    Case('experts_32', 'balanced', balanced_counts(4096, 32)),
    Case('hot_50', 'hot=50%', hot_expert_counts(4096, 8, 0.5)),
    Case('hot_875', 'hot=87.5%', hot_expert_counts(4096, 8, 0.875)),
    Case('model_small', 'balanced', balanced_counts(1024, 8), 128, 512),
    Case('model_large', 'balanced', balanced_counts(1024, 8), 512, 2048),
]


def get_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def read_metadata(compiled: Any) -> dict[str, Any]:
    metadata = getattr(compiled, 'metadata', None)
    return {
        'regs': get_attr(compiled, ['n_regs', 'num_regs']),
        'spills': get_attr(compiled, ['n_spills', 'num_spills']),
        'threads': get_attr(compiled, ['n_max_threads', 'max_threads']),
        'shared': get_attr(metadata, ['shared', 'shared_memory', 'shared_mem']),
    }


def format_status(exc: Exception) -> str:
    message = str(exc).splitlines()[0]
    if 'out of resource: shared memory' in message:
        return 'oom_shared'
    if 'OutOfResources' in type(exc).__name__:
        return 'out_of_resources'
    if message:
        return f'error:{type(exc).__name__}: {message[:96]}'
    return f'error:{type(exc).__name__}'


def make_expert_offsets(expert_counts: tuple[int, ...], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for count in expert_counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def warmup_case(case: Case, device: torch.device) -> dict[str, Any]:
    dtype = torch.bfloat16
    expert_offsets = make_expert_offsets(case.expert_counts, device)
    x_grouped = torch.empty(
        (case.num_assignments + 1, case.d_model),
        dtype=dtype,
        device=device,
    )
    w = torch.empty(
        (case.n_experts, case.d_model, case.d_ff),
        dtype=dtype,
        device=device,
    )
    out = torch.empty(
        (case.num_assignments + 1, case.d_ff),
        dtype=dtype,
        device=device,
    )

    max_m_tiles = triton.cdiv(case.expert_capacity, BLOCK_M)
    n_tiles = triton.cdiv(case.d_ff, BLOCK_N)
    grid = (case.n_experts * max_m_tiles, n_tiles)
    active_m_tiles = sum(triton.cdiv(count, BLOCK_M) for count in case.expert_counts)
    launched_m_tiles = grid[0]

    compiled = moe_grouped_expert_gemm_fwd_v1_fused_kernel.warmup(
        x_grouped,
        w,
        expert_offsets,
        out,
        x_grouped.stride(0),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        MAX_M_TILES=max_m_tiles,
        N=case.d_ff,
        K=case.d_model,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        grid=grid,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    init_handles = getattr(compiled, '_init_handles', None)
    if init_handles is not None:
        init_handles()

    return {
        'case': case.name,
        'routing': case.distribution,
        'M': case.num_assignments,
        'E': case.n_experts,
        'capacity': case.expert_capacity,
        'K': case.d_model,
        'N': case.d_ff,
        'grid': f'{grid[0]}x{grid[1]}',
        'active_M': active_m_tiles,
        'launched_M': launched_m_tiles,
        'overlaunch': f'{launched_m_tiles / active_m_tiles:.2f}x',
        **read_metadata(compiled),
        'status': 'ok',
    }


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = [
        'case',
        'routing',
        'M',
        'E',
        'capacity',
        'K',
        'N',
        'grid',
        'active_M',
        'launched_M',
        'overlaunch',
        'regs',
        'spills',
        'threads',
        'shared',
        'status',
    ]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ''))) for row in rows))
        for column in columns
    }

    print(
        f'config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, '
        f'num_warps={NUM_WARPS}, num_stages={NUM_STAGES}'
    )
    print('  '.join(f'{column:<{widths[column]}}' for column in columns))
    print('  '.join('-' * widths[column] for column in columns))
    for row in rows:
        print('  '.join(f"{str(row.get(column, '')):<{widths[column]}}" for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description='Show grouped expert GEMM Triton kernel metadata.')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to inspect Triton kernel metadata.')

    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')

    rows = []
    for case in CASES:
        try:
            rows.append(warmup_case(case, device))
        except Exception as exc:
            rows.append(
                {
                    'case': case.name,
                    'routing': case.distribution,
                    'M': case.num_assignments,
                    'E': case.n_experts,
                    'capacity': case.expert_capacity,
                    'K': case.d_model,
                    'N': case.d_ff,
                    'status': format_status(exc),
                }
            )

    print_rows(rows)


if __name__ == '__main__':
    main()
