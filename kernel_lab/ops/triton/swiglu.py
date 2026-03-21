from __future__ import annotations

from kernel_lab.ops.registry import OperatorImplementation, register


def swiglu_triton(*args, **kwargs):
    raise NotImplementedError(
        "Implement the Triton SwiGLU kernel here, then mark the backend as available."
    )


register(
    OperatorImplementation(
        name="swiglu",
        backend="triton",
        fn=swiglu_triton,
        is_available=lambda: False,
        description="Template slot for the Triton SwiGLU kernel.",
    )
)

