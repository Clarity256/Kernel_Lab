from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from kernel_lab.ops.common.utils import benchmark_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a toy attention baseline.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    q = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype)

    fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False)
    stats = benchmark_call(fn, warmup=args.warmup, iters=args.iters, device=device)

    print("op=attention_toy backend=torch_sdpa")
    print(
        f"shape=({args.batch}, {args.heads}, {args.seq_len}, {args.head_dim}) device={device} dtype={dtype}"
    )
    print(f"avg_ms={stats['avg_ms']:.4f} min_ms={stats['min_ms']:.4f} max_ms={stats['max_ms']:.4f}")


if __name__ == "__main__":
    main()

