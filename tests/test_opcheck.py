import torch

from kernel_lab.ops import registry


def test_registry_lists_expected_study_ops() -> None:
    expected = {"attention_toy", "rmsnorm", "rope", "softmax", "swiglu"}
    assert expected.issubset(set(registry.list_ops()))


def test_registry_falls_back_to_reference() -> None:
    x = torch.randn(2, 16, 32)

    expected = registry.run("softmax", x, backend="reference")
    actual = registry.run("softmax", x, preferred=("cuda", "triton", "reference"))

    torch.testing.assert_close(actual, expected)

