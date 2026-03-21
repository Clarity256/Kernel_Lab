from __future__ import annotations

import importlib.util

import torch

from kernel_lab.ops.registry import OperatorImplementation, register


if importlib.util.find_spec("kernel_lab_cuda") is not None:
    import kernel_lab_cuda as _C
else:
    _C = None


def _cuda_extension_available() -> bool:
    return _C is not None and torch.cuda.is_available()


def softmax_cuda(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("CUDA extension is not built. Run `python setup.py build_ext --inplace`.")
    return _C.softmax_forward(x, dim)


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("CUDA extension is not built. Run `python setup.py build_ext --inplace`.")
    return _C.rmsnorm_forward(x, weight, eps)


register(
    OperatorImplementation(
        name="softmax",
        backend="cuda",
        fn=softmax_cuda,
        is_available=_cuda_extension_available,
        description="CUDA extension entry point for softmax.",
    )
)

register(
    OperatorImplementation(
        name="rmsnorm",
        backend="cuda",
        fn=rmsnorm_cuda,
        is_available=_cuda_extension_available,
        description="CUDA extension entry point for RMSNorm.",
    )
)

