import numpy as np

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

# def test_reverse_add() -> None:
#     pass
