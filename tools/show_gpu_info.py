import argparse
from typing import Any

import torch


def format_bytes(nbytes: int | None) -> str:
    if nbytes is None:
        return "n/a"

    value = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{nbytes} B"


def print_kv_section(title: str, rows: list[tuple[str, Any]]) -> None:
    print(title)
    print("-" * len(title))

    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")


def print_device_properties(device_index: int) -> None:
    props = torch.cuda.get_device_properties(device_index)

    rows = [
        ("name", props.name),
        ("compute capability", f"{props.major}.{props.minor}"),
        ("SM count", props.multi_processor_count),
        ("warp size", getattr(props, "warp_size", "n/a")),
        ("total memory", format_bytes(props.total_memory)),
        ("L2 cache", format_bytes(getattr(props, "l2_cache_size", None))),
        ("regs / SM", getattr(props, "regs_per_multiprocessor", "n/a")),
        ("regs / block", getattr(props, "regs_per_block", "n/a")),
        ("shared memory / SM", format_bytes(getattr(props, "shared_memory_per_multiprocessor", None))),
        ("shared memory / block", format_bytes(getattr(props, "shared_memory_per_block", None))),
        ("max threads / SM", getattr(props, "max_threads_per_multiprocessor", "n/a")),
        ("max threads / block", getattr(props, "max_threads_per_block", "n/a")),
    ]

    print_kv_section("Device", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show CUDA device properties.")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to inspect GPU device properties.")

    torch.cuda.set_device(args.device)
    print_device_properties(args.device)


if __name__ == "__main__":
    main()
