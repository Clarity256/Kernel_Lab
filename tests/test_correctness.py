import pytest
import torch

from kernel_lab.ops import registry
from kernel_lab.ops.common.shapes import get_shape


@pytest.mark.parametrize("shape_name", ["tiny", "llm_decode"])
def test_softmax_reference_matches_torch(shape_name: str) -> None:
    shape = get_shape(shape_name)
    x = torch.randn(shape.batch, shape.seq_len, shape.hidden_size)

    expected = torch.softmax(x, dim=-1)
    actual = registry.run("softmax", x, backend="reference")

    torch.testing.assert_close(actual, expected)


def test_rmsnorm_reference_matches_formula() -> None:
    shape = get_shape("tiny")
    x = torch.randn(shape.batch, shape.seq_len, shape.hidden_size)
    weight = torch.randn(shape.hidden_size)

    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight
    actual = registry.run("rmsnorm", x, weight, backend="reference")

    torch.testing.assert_close(actual, expected)


def test_optional_backends_match_reference_if_available() -> None:
    for op_name in ("softmax", "rmsnorm"):
        optional_backends = [
            backend for backend in registry.available_backends(op_name) if backend != "reference"
        ]
        for backend in optional_backends:
            device = torch.device("cuda" if backend in {"triton", "cuda"} else "cpu")
            x = torch.randn(2, 64, 128, device=device)
            op_args = (x,) if op_name == "softmax" else (x, torch.randn(128, device=device))

            expected = registry.run(op_name, *op_args, backend="reference")
            actual = registry.run(op_name, *op_args, backend=backend)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
