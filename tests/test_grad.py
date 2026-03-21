import torch
from torch.autograd import gradcheck

from kernel_lab.ops import registry


def test_softmax_reference_gradcheck() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    assert gradcheck(lambda inp: registry.run("softmax", inp, backend="reference"), (x,))


def test_rmsnorm_reference_gradcheck() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    weight = torch.randn(8, dtype=torch.double, requires_grad=True)
    assert gradcheck(lambda inp, w: registry.run("rmsnorm", inp, w, backend="reference"), (x, weight))

