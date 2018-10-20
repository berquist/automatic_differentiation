import numpy as np

from dual import Dual, sin, log, exp


def test_dual_lift():
    a = -3
    b = Dual(-3, 0)
    assert Dual._lift(a) == b


def test_dual_add():
    a = Dual(1, 1)
    b = Dual(2, 3)
    c = Dual(3, 4)
    assert a + b == c


def test_dual_sub():
    a = Dual(1, 1)
    b = Dual(2, 3)
    c = Dual(-1, -2)
    assert a - b == c


def test_dual_mul():
    a = Dual(1, 1)
    b = Dual(2, 3)
    c = Dual(2, 3)
    assert a * b == c


def f(x, y):
    return sin(log(7 * x) + exp(y)) + 9


if __name__ == "__main__":
    x = np.e
    y = np.pi
    print(f(x, y))
    print(f(Dual._lift(x), Dual._lift(y)))
    print(f(Dual(x, 1), Dual(y, 1)))
