import math

import autograd
import autograd.numpy as np
import torch
from pytest import approx

from autodiff.reverse import Var


def test_reverse_example() -> None:
    """Cribbed from https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py"""

    x = Var(0.5)
    y = Var(4.2)
    z = x * y + x.sin()
    z.grad_value = 1.0

    thresh = 1e-15

    assert abs(z.value - 2.579425538604203) <= thresh
    assert (x.grad() - (y.value + np.cos(x.value))) <= thresh
    assert (y.grad() - x.value) <= thresh


def test_sin() -> None:
    """Make sure the derivative of the sine is consistent between
    autodiff.reverse, pytorch, and autograd.
    """
    y = 4.2
    z = math.sin(y)
    ref = math.cos(y)

    # autodiff.reverse
    y_r = Var(y)
    z_r = y_r.sin()
    z_r.grad_value = 1.0

    # pytorch
    y_t = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    z_t = y_t.sin()
    z_t.backward()

    # autograd
    def f(a):
        return np.sin(a)

    assert y_r.grad() == approx(ref)
    assert y_t.grad.item() == approx(ref)
    assert autograd.grad(f)(y) == approx(ref)
