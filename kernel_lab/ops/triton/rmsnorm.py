from __future__ import annotations

from kernel_lab.ops.registry import OperatorImplementation, register


def rmsnorm_triton(*args, **kwargs):
    raise NotImplementedError(
        "Implement the Triton RMSNorm kernel here, then mark the backend as available."
    )


register(
    OperatorImplementation(
        name="rmsnorm",
        backend="triton",
        fn=rmsnorm_triton,
        is_available=lambda: False,
        description="Template slot for the Triton RMSNorm kernel.",
    )
)

