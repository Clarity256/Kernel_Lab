from __future__ import annotations

import time

import torch


def preferred_backends_for_device(device: torch.device) -> tuple[str, ...]:
    if device.type == "cuda":
        return ("cuda", "triton", "reference")
    return ("reference",)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_call(
    fn,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    return {
        "avg_ms": avg_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
