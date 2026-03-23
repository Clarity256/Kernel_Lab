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


_MAX_BLOCK_SIZE = 1024


if triton is not None:

    @triton.jit
    def _naive_softmax_kernel(
        output_ptr,
        input_ptr,
        output_row_stride,
        input_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_input_ptr = input_ptr + row_id * input_row_stride
        row_output_ptr = output_ptr + row_id * output_row_stride

        row_max = -float("inf")
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(x, axis=0))

        row_denom = 0.0
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            row_denom += tl.sum(tl.exp(x - row_max), axis=0)

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            y = tl.exp(x - row_max) / row_denom
            tl.store(row_output_ptr + cols, y, mask=mask)


    @triton.jit
    def _online_softmax_kernel(
        output_ptr,
        input_ptr,
        output_row_stride,
        input_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_id = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_input_ptr = input_ptr + row_id * input_row_stride
        row_output_ptr = output_ptr + row_id * output_row_stride

        row_max = -float("inf")
        row_denom = 0.0
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            block_max = tl.max(x, axis=0)
            block_denom = tl.sum(tl.exp(x - block_max), axis=0)

            next_max = tl.maximum(row_max, block_max)
            row_denom = row_denom * tl.exp(row_max - next_max) + block_denom * tl.exp(
                block_max - next_max
            )
            row_max = next_max

        for start in tl.range(0, n_cols, BLOCK_SIZE):
            cols = start + offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            y = tl.exp(x - row_max) / row_denom
            tl.store(row_output_ptr + cols, y, mask=mask)


def _triton_available() -> bool:
    return triton is not None and triton_runtime_available()


def _num_warps_for(block_size: int) -> int:
    if block_size <= 128:
        return 4
    if block_size <= 512:
        return 8
    return 16


def _validate_softmax_inputs(x: torch.Tensor, dim: int) -> None:
    if x.device.type != "cuda":
        raise ValueError("Triton softmax requires a CUDA tensor input.")
    if dim not in (-1, x.ndim - 1):
        raise NotImplementedError("This study kernel currently supports softmax on the last dimension.")


def _launch_softmax_kernel(
    kernel,
    x: torch.Tensor,
) -> torch.Tensor:
    n_cols = x.shape[-1]
    block_size = min(_MAX_BLOCK_SIZE, triton.next_power_of_2(n_cols))
    x_2d = x.contiguous().view(-1, n_cols)
    y_2d = torch.empty_like(x_2d)

    grid = (x_2d.shape[0],)
    kernel[grid](
        y_2d,
        x_2d,
        y_2d.stride(0),
        x_2d.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps_for(block_size),
    )
    return y_2d.view_as(x)


def naive_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if not _triton_available():
        raise RuntimeError("Triton runtime is unavailable. Install Triton on a Linux + CUDA host.")

    _validate_softmax_inputs(x, dim)
    return _launch_softmax_kernel(_naive_softmax_kernel, x)


def online_softmax(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if not _triton_available():
        raise RuntimeError("Triton runtime is unavailable. Install Triton on a Linux + CUDA host.")

    _validate_softmax_inputs(x, dim)
    return _launch_softmax_kernel(_online_softmax_kernel, x)


def softmax_triton(
    x: torch.Tensor,
    dim: int = -1,
    algorithm: str = "online",
) -> torch.Tensor:
    if algorithm == "online":
        return online_softmax(x, dim=dim)
    if algorithm == "naive":
        return naive_softmax(x, dim=dim)
    raise ValueError("algorithm must be one of: 'online', 'naive'.")


register(
    OperatorImplementation(
        name="softmax",
        backend="triton",
        fn=softmax_triton,
        is_available=_triton_available,
        description="Triton softmax with naive and online study implementations.",
    )
)
