import numpy as np

from autodiff.dualmat import DualMat2D


def test_dualmat_lift() -> None:
    a = -3
    b = DualMat2D.lift(a)
    # pylint: disable=C0326
    c = np.array([[ -3,  0],
                  [  0, -3]], dtype=int)
    np.testing.assert_equal(b.mat, c)


def test_dualmat_init() -> None:
    a = -3
    b = 1
    # pylint: disable=C0326
    c = np.array([[ -3,  1],
                  [  0, -3]], dtype=int)
    d = DualMat2D.from_vals(a, b)
    np.testing.assert_equal(d.mat, c)


def test_dualmat_add() -> None:
    a = DualMat2D.from_vals(-3, 1)
    b = DualMat2D.from_vals(-5, 2)
    c_ = DualMat2D.from_vals(-8, 3)
    # pylint: disable=C0326
    refmat = np.array([[-8,  3],
                       [ 0, -8]], dtype=int)
    c = a + b
    np.testing.assert_equal(c.first, c_.first)
    np.testing.assert_equal(c.second, c_.second)
    np.testing.assert_equal(c.mat, refmat)
    np.testing.assert_equal(c_.mat, refmat)


def test_dualmat_sub() -> None:
    a = DualMat2D.from_vals(-3, 1)
    b = DualMat2D.from_vals(-5, 2)
    c_ = DualMat2D.from_vals(2, -1)
    # pylint: disable=C0326
    refmat = np.array([[ 2, -1],
                       [ 0,  2]], dtype=int)
    c = a - b
    np.testing.assert_equal(c.first, c_.first)
    np.testing.assert_equal(c.second, c_.second)
    np.testing.assert_equal(c.mat, refmat)
    np.testing.assert_equal(c_.mat, refmat)


def test_dualmat_mul() -> None:
    a = DualMat2D.from_vals(-3, 1)
    b = DualMat2D.from_vals(-5, 2)
    c_ = DualMat2D.from_vals(15, -11)
    # pylint: disable=C0326
    refmat = np.array([[15, -11],
                       [ 0,  15]], dtype=int)
    c = a * b
    np.testing.assert_equal(c.first, c_.first)
    np.testing.assert_equal(c.second, c_.second)
    np.testing.assert_equal(c.mat, refmat)
    np.testing.assert_equal(c_.mat, refmat)


def test_dualmat_div() -> None:
    a = DualMat2D.from_vals(-3, 1)
    b = DualMat2D.from_vals(-5, 2)
    c_ = DualMat2D.from_vals(0.6, 0.04)
    # pylint: disable=C0326
    refmat = np.array([[0.60, 0.04],
                       [0.00, 0.60]])
    c = a / b
    np.testing.assert_almost_equal(c.first, c_.first)
    np.testing.assert_almost_equal(c.second, c_.second)
    np.testing.assert_almost_equal(c.mat, refmat)
    np.testing.assert_almost_equal(c_.mat, refmat)
