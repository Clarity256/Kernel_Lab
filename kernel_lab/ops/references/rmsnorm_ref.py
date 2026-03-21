from __future__ import annotations

import torch

from kernel_lab.ops.registry import OperatorImplementation, register


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(variance + eps)
    return x_hat * weight


register(
    OperatorImplementation(
        name="rmsnorm",
        backend="reference",
        fn=rmsnorm_reference,
        is_available=lambda: True,
        description="Readable Torch RMSNorm baseline.",
    )
)

