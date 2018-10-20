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
    a = Dual(2, 3)
    b = Dual(4, 5)
    # (2 * 5) + (3 * 4) = 10 + 12 = 22
    c = Dual(8, 22)
    assert a * b == c


def f_1d(x):
    return (3.67 * (x ** 3)) - 2


def f(x, y):
    return sin(log(7 * x) + exp(y)) + 9


if __name__ == "__main__":
    # x = np.e
    # y = np.pi
    # print(f(x, y))
    # print(f(Dual._lift(x), Dual._lift(y)))
    # d1 = f(Dual(x, 1), Dual(y, 1))
    # print(d1)

    x = 0.9
    d0 = Dual(x, 0)
    d1 = Dual(x, 1)
    print(f_1d(x))
    print(f_1d(d0))
    print(f_1d(d1))
