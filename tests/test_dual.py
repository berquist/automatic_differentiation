from dual import Dual


def test_dual_lift():
    a = -3
    b = Dual(-3, 0)
    assert Dual._lift(a) == b


def test_dual_const_add():
    a = Dual(2, 3)
    b = 5
    c = Dual(7, 3)
    assert a + b == c


def test_dual_dual_add():
    a = Dual(1, 1)
    b = Dual(2, 3)
    c = Dual(3, 4)
    assert a + b == c


def test_dual_const_sub():
    a = Dual(2, 3)
    b = 5
    c = Dual(-3, 3)
    assert a - b == c


def test_dual_dual_sub():
    a = Dual(1, 1)
    b = Dual(2, 3)
    c = Dual(-1, -2)
    assert a - b == c


def test_dual_const_mul():
    a = Dual(2, 3)
    b = 5
    # (2 * 0) + (5 * 3) = 0 + 15 = 15
    c = Dual(10, 15)
    assert a * b == c


def test_dual_dual_mul():
    a = Dual(2, 3)
    b = Dual(4, 5)
    # (2 * 5) + (3 * 4) = 10 + 12 = 22
    c = Dual(8, 22)
    assert a * b == c
