from __future__ import annotations

import torch

from kernel_lab.ops.registry import OperatorImplementation, register
from kernel_lab.ops.triton._common import triton_runtime_available

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        row_ptr = x_ptr + row_idx * row_stride + offsets
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean_square = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        y = x * inv_rms * weight

        tl.store(y_ptr + row_idx * row_stride + offsets, y, mask=mask)


def _triton_available() -> bool:
    return triton is not None and triton_runtime_available()


def _num_warps_for(block_size: int) -> int:
    if block_size <= 256:
        return 4
    if block_size <= 1024:
        return 8
    return 16


def _validate_rmsnorm_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.device.type != "cuda":
        raise ValueError("Triton RMSNorm requires a CUDA tensor input.")
    if weight.device != x.device:
        raise ValueError("weight must live on the same device as x.")
    if weight.ndim != 1:
        raise ValueError("weight must be a 1D tensor.")
    if x.shape[-1] != weight.numel():
        raise ValueError("The last dimension of x must match weight.numel().")


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not _triton_available():
        raise RuntimeError("Triton runtime is unavailable. Install Triton on a Linux + CUDA host.")

    _validate_rmsnorm_inputs(x, weight)

    hidden_size = x.shape[-1]
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise ValueError("This study kernel expects hidden_size <= 65536.")

    x_2d = x.contiguous().view(-1, hidden_size)
    weight = weight.contiguous()
    y_2d = torch.empty_like(x_2d)

    grid = (x_2d.shape[0],)
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        y_2d,
        x_2d.stride(0),
        hidden_size,
        float(eps),
        BLOCK_SIZE=block_size,
        num_warps=_num_warps_for(block_size),
    )
    return y_2d.view_as(x)


register(
    OperatorImplementation(
        name="rmsnorm",
        backend="triton",
        fn=rmsnorm_triton,
        is_available=_triton_available,
        description="Minimal Triton RMSNorm kernel for framework study.",
    )
)
