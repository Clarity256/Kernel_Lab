from __future__ import annotations

from kernel_lab.ops.registry import OperatorImplementation, register


def softmax_triton(*args, **kwargs):
    raise NotImplementedError(
        "Implement the Triton softmax kernel here, then mark the backend as available."
    )


register(
    OperatorImplementation(
        name="softmax",
        backend="triton",
        fn=softmax_triton,
        is_available=lambda: False,
        description="Template slot for the Triton softmax kernel.",
    )
)

