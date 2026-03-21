from __future__ import annotations

import torch

from kernel_lab.ops.registry import OperatorImplementation, register


def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


register(
    OperatorImplementation(
        name="softmax",
        backend="reference",
        fn=softmax_reference,
        is_available=lambda: True,
        description="Torch softmax baseline used for correctness and benchmarking.",
    )
)

