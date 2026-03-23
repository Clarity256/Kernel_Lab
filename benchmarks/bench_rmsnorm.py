from __future__ import annotations

import argparse

import torch

from kernel_lab.ops import registry
from kernel_lab.ops.common.shapes import get_shape
from kernel_lab.ops.common.utils import benchmark_call, preferred_backends_for_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm across available backends.")
    parser.add_argument("--backend", default="auto", choices=["auto", "reference", "triton", "cuda"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shape", default="tiny")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = get_shape(args.shape)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    x = torch.randn(shape.batch, shape.seq_len, shape.hidden_size, device=device, dtype=dtype)
    weight = torch.randn(shape.hidden_size, device=device, dtype=dtype)
    if args.backend == "auto":
        preferred = preferred_backends_for_device(device)
        fn = lambda: registry.run("rmsnorm", x, weight, preferred=preferred)
        backend_name = registry.get("rmsnorm", preferred=preferred).backend
    else:
        fn = lambda: registry.run("rmsnorm", x, weight, backend=args.backend)
        backend_name = args.backend

    stats = benchmark_call(fn, warmup=args.warmup, iters=args.iters, device=device)
    print(f"op=rmsnorm backend={backend_name} shape={shape} device={device} dtype={dtype}")
    print(f"avg_ms={stats['avg_ms']:.4f} min_ms={stats['min_ms']:.4f} max_ms={stats['max_ms']:.4f}")


if __name__ == "__main__":
    main()
