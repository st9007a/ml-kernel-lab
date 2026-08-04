import argparse
from dataclasses import dataclass
from typing import Any

import torch
import triton

from ml_kernel_lab.kernel.triton_kernel.moe import moe_grouped_expert_gemm_fwd_v1_fused_kernel


@dataclass(frozen=True)
class KernelConfig:
    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 3


CONFIGS = [
    KernelConfig('small', 64, 128, 32, 4),
    KernelConfig('production_k32', 128, 128, 32, 8),
    KernelConfig('production_k64', 128, 128, 64, 8),
]


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
    Case('production_small', 'balanced', balanced_counts(4096, 8), 1024, 4096),
    Case('production_medium', 'balanced', balanced_counts(4096, 8), 2048, 8192),
    Case('production_large_4096', 'balanced', balanced_counts(4096, 8), 4096, 14336),
    Case('production_large_8192', 'balanced', balanced_counts(8192, 8), 4096, 14336),
    Case('production_large_16384', 'balanced', balanced_counts(16384, 8), 4096, 14336),
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


def warmup_case(
    case: Case,
    config: KernelConfig,
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.bfloat16
    expert_offsets = make_expert_offsets(case.expert_counts, device)
    # warmup compiles without executing, so full production-sized storage is unnecessary.
    x_grouped = torch.empty((1,), dtype=dtype, device=device)
    w = torch.empty((1,), dtype=dtype, device=device)
    out = torch.empty((1,), dtype=dtype, device=device)

    max_m_tiles = triton.cdiv(case.expert_capacity, config.block_m)
    n_tiles = triton.cdiv(case.d_ff, config.block_n)
    grid = (case.n_experts * max_m_tiles, n_tiles)
    active_m_tiles = sum(
        triton.cdiv(count, config.block_m)
        for count in case.expert_counts
    )
    launched_m_tiles = grid[0]

    compiled = moe_grouped_expert_gemm_fwd_v1_fused_kernel.warmup(
        x_grouped,
        w,
        expert_offsets,
        out,
        case.d_model,
        case.d_model * case.d_ff,
        case.d_ff,
        case.d_ff,
        M=case.num_assignments,
        N=case.d_ff,
        K=case.d_model,
        N_EXPERTS=case.n_experts,
        EXPERT_CAPACITY=case.expert_capacity,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        grid=grid,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    init_handles = getattr(compiled, '_init_handles', None)
    if init_handles is not None:
        init_handles()

    kernel_metadata = read_metadata(compiled)
    cta_threads = config.num_warps * 32
    max_threads = kernel_metadata.pop('threads')
    resident_ctas = max_threads // cta_threads if max_threads is not None else None

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
        **kernel_metadata,
        'cta_threads': cta_threads,
        'max_threads': max_threads,
        'resident_ctas': resident_ctas,
        'status': 'ok',
    }


def print_rows(config: KernelConfig, rows: list[dict[str, Any]]) -> None:
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
        'cta_threads',
        'max_threads',
        'resident_ctas',
        'shared',
        'status',
    ]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ''))) for row in rows))
        for column in columns
    }

    print(
        f'config={config.name}: BLOCK_M={config.block_m}, BLOCK_N={config.block_n}, '
        f'BLOCK_K={config.block_k}, num_warps={config.num_warps}, '
        f'num_stages={config.num_stages}'
    )
    print('  '.join(f'{column:<{widths[column]}}' for column in columns))
    print('  '.join('-' * widths[column] for column in columns))
    for row in rows:
        print('  '.join(f"{str(row.get(column, '')):<{widths[column]}}" for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description='Show grouped expert GEMM Triton kernel metadata.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--config',
        choices=['all', *(config.name for config in CONFIGS)],
        default='all',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to inspect Triton kernel metadata.')

    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')

    selected_configs = CONFIGS
    if args.config != 'all':
        selected_configs = [config for config in CONFIGS if config.name == args.config]

    for config_idx, config in enumerate(selected_configs):
        rows = []
        for case in CASES:
            try:
                rows.append(warmup_case(case, config, device))
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

        if config_idx:
            print()
        print_rows(config, rows)


if __name__ == '__main__':
    main()
