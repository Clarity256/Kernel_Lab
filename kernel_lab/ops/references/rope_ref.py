from __future__ import annotations

import torch

from kernel_lab.ops.registry import OperatorImplementation, register


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)


def rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


register(
    OperatorImplementation(
        name="rope",
        backend="reference",
        fn=rope_reference,
        is_available=lambda: True,
        description="Torch rotary embedding baseline.",
    )
)

