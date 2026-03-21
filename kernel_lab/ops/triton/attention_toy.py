from __future__ import annotations

from kernel_lab.ops.registry import OperatorImplementation, register


def attention_toy_triton(*args, **kwargs):
    raise NotImplementedError(
        "Implement the toy Triton attention kernel here, then mark the backend as available."
    )


register(
    OperatorImplementation(
        name="attention_toy",
        backend="triton",
        fn=attention_toy_triton,
        is_available=lambda: False,
        description="Template slot for a study-oriented Triton attention kernel.",
    )
)

