import argparse
import math
from dataclasses import dataclass
from typing import Any

import torch
import triton

from ml_kernel_lab.kernel.triton_kernel.flash_attn_v1 import flash_attention_v1_fwd_fused_kernel


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    n_heads: int
    seq_len: int
    head_dim: int
    is_causal: bool = False


@dataclass(frozen=True)
class Config:
    q_tile_size: int
    k_tile_size: int
    num_warps: int
    num_stages: int


CASES = [
    Case("seq_len_128", 1, 8, 128, 128),
    Case("seq_len_256", 1, 8, 256, 128),
    Case("seq_len_512", 1, 8, 512, 128),
    Case("seq_len_1024", 1, 8, 1024, 128),
    Case("causal_seq_len_128", 1, 8, 128, 128, True),
    Case("causal_seq_len_256", 1, 8, 256, 128, True),
    Case("causal_seq_len_512", 1, 8, 512, 128, True),
    Case("causal_seq_len_1024", 1, 8, 1024, 128, True),
    Case("batch_size_2", 2, 8, 512, 128),
    Case("batch_size_4", 4, 8, 512, 128),
    Case("batch_size_8", 8, 8, 512, 128),
    Case("head_dim_64", 1, 8, 512, 64),
]


CONFIGS = [
    Config(32, 32, 4, 3),
    Config(32, 64, 4, 3),
    Config(64, 64, 4, 3),
    Config(64, 64, 8, 3),
    Config(128, 32, 8, 3),
    Config(128, 64, 8, 3),
    Config(128, 128, 8, 3),
]


def get_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def read_metadata(compiled: Any) -> dict[str, Any]:
    metadata = getattr(compiled, "metadata", None)
    return {
        "regs": get_attr(compiled, ["n_regs", "num_regs"]),
        "spills": get_attr(compiled, ["n_spills", "num_spills"]),
        "threads": get_attr(compiled, ["n_max_threads", "max_threads"]),
        "shared": get_attr(metadata, ["shared", "shared_memory", "shared_mem"]),
    }


def format_status(exc: Exception) -> str:
    message = str(exc).splitlines()[0]
    if "out of resource: shared memory" in message:
        return "oom_shared"
    if "OutOfResources" in type(exc).__name__:
        return "out_of_resources"
    return f"error:{type(exc).__name__}"


def warmup_case(case: Case, config: Config, device: torch.device) -> dict[str, Any]:
    dtype = torch.bfloat16
    shape = (case.batch_size, case.n_heads, case.seq_len, case.head_dim)
    q = torch.empty(shape, dtype=dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = torch.empty_like(q)

    n_q_tiles = triton.cdiv(case.seq_len, config.q_tile_size)
    n_kv_tiles = triton.cdiv(case.seq_len, config.k_tile_size)
    grid = (n_q_tiles, case.batch_size * case.n_heads)

    compiled = flash_attention_v1_fwd_fused_kernel.warmup(
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_kv_tiles,
        case.n_heads,
        case.seq_len,
        case.head_dim,
        1.0 / math.sqrt(case.head_dim),
        config.q_tile_size,
        config.k_tile_size,
        triton.next_power_of_2(case.head_dim),
        triton.next_power_of_2(config.q_tile_size),
        triton.next_power_of_2(config.k_tile_size),
        case.is_causal,
        grid=grid,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    init_handles = getattr(compiled, "_init_handles", None)
    if init_handles is not None:
        init_handles()

    return {
        "case": case.name,
        "mode": "causal" if case.is_causal else "plain",
        "B": case.batch_size,
        "H": case.n_heads,
        "N": case.seq_len,
        "D": case.head_dim,
        "Q": config.q_tile_size,
        "KV": config.k_tile_size,
        "warps": config.num_warps,
        "stages": config.num_stages,
        "grid": f"{grid[0]}x{grid[1]}",
        **read_metadata(compiled),
        "status": "ok",
    }


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = [
        "case",
        "mode",
        "B",
        "H",
        "N",
        "D",
        "Q",
        "KV",
        "warps",
        "stages",
        "grid",
        "regs",
        "spills",
        "threads",
        "shared",
        "status",
    ]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }

    print("  ".join(f"{column:<{widths[column]}}" for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(f"{str(row.get(column, '')):<{widths[column]}}" for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show FlashAttention v1 Triton kernel metadata.")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to inspect Triton kernel metadata.")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    rows = []
    for case in CASES:
        for config in CONFIGS:
            try:
                rows.append(warmup_case(case, config, device))
            except Exception as exc:
                rows.append(
                    {
                        "case": case.name,
                        "mode": "causal" if case.is_causal else "plain",
                        "B": case.batch_size,
                        "H": case.n_heads,
                        "N": case.seq_len,
                        "D": case.head_dim,
                        "Q": config.q_tile_size,
                        "KV": config.k_tile_size,
                        "warps": config.num_warps,
                        "stages": config.num_stages,
                        "grid": "",
                        "status": format_status(exc),
                    }
                )

    print_rows(rows)


if __name__ == "__main__":
    main()
