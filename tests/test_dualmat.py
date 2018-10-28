import numpy as np

from dualmat import DualMat2D


def test_dualmat_lift():
    a = -3
    b = DualMat2D._lift(a)
    c = np.array([[ -3,  0],
                  [  0, -3]], dtype=int)
    np.testing.assert_equal(b.mat, c)


def test_dualmat_init():
    a = -3
    b = 1
    c = np.array([[ -3,  1],
                  [  0, -3]], dtype=int)
    d = DualMat2D(a, b)
    np.testing.assert_equal(d.mat, c)