from functools import partial

import numpy as np
import sympy as sy

from autodiff.dual import Dual

from autodiff.tests.common import f_1d, f_2d, f_2d_sympy


def test_f_mul() -> None:
    x = 0.9
    xs = sy.Symbol('x')
    values = {xs: x}

    def f(x):
        return (7 * x) + 2

    f_sympy_symbolic = f(xs)
    df_dx_sympy_symbolic = f_sympy_symbolic.diff(xs)
    d2f_dx2_sympy_symbolic = df_dx_sympy_symbolic.diff(xs)

    f_sympy = f_sympy_symbolic.evalf(subs=values)
    df_dx_sympy = df_dx_sympy_symbolic.evalf(subs=values)
    d2f_dx2_sympy = d2f_dx2_sympy_symbolic.evalf(subs=values)

    print(f_sympy_symbolic)
    print(f_sympy)
    print(df_dx_sympy_symbolic)
    print(df_dx_sympy)
    print(d2f_dx2_sympy_symbolic)
    print(d2f_dx2_sympy)

    f1 = f(Dual(x, 1))
    print(f1)

    decimal = 14

    np.testing.assert_almost_equal(f1.first, f_sympy, decimal=decimal)
    np.testing.assert_almost_equal(f1.second, df_dx_sympy, decimal=decimal)


def test_f_1d() -> None:
    x = 0.9
    xs = sy.Symbol('x')
    values = {xs: x}

    fs = f_1d(xs)
    f_sympy = fs.evalf(subs=values)
    df_dx_sympy_symbolic = fs.diff(xs)
    d2f_dx_dx_sympy_symbolic = df_dx_sympy_symbolic.diff(xs)
    df_dx_sympy = df_dx_sympy_symbolic.evalf(subs=values)
    d2f_dx_dx_sympy = d2f_dx_dx_sympy_symbolic.evalf(subs=values)

    print('-' * 70)
    print(fs)
    print(f_sympy)
    print('-' * 70)
    print(df_dx_sympy_symbolic)
    print(df_dx_sympy)
    print('-' * 70)
    print(d2f_dx_dx_sympy_symbolic)
    print(d2f_dx_dx_sympy)

    d0 = Dual(x, 0)
    d1 = Dual(x, 1)
    result_1d_x = f_1d(x)
    result_1d_d0 = f_1d(d0)
    result_1d_d1 = f_1d(d1)
    assert isinstance(result_1d_x, float)
    assert isinstance(result_1d_d0, Dual)
    assert isinstance(result_1d_d1, Dual)

    print('-' * 70)
    print(result_1d_x)
    print(result_1d_d0)
    print(result_1d_d1)

    decimal = 14

    np.testing.assert_almost_equal(result_1d_x, f_sympy,
                                   decimal=decimal)
    np.testing.assert_almost_equal(result_1d_d0.first, f_sympy,
                                   decimal=decimal)
    np.testing.assert_almost_equal(result_1d_d0.second, 0.0,
                                   decimal=decimal)
    np.testing.assert_almost_equal(result_1d_d1.first, f_sympy,
                                   decimal=decimal)
    np.testing.assert_almost_equal(result_1d_d1.second, df_dx_sympy,
                                   decimal=decimal)


def test_f_2d() -> None:
    x = np.e
    y = np.pi

    xs, ys = sy.symbols('x y')
    f_sympy_symbolic = f_2d_sympy(xs, ys)
    df_dx_sympy_symbolic = f_sympy_symbolic.diff(xs)
    df_dy_sympy_symbolic = f_sympy_symbolic.diff(ys)
    d2f_dx_dx_sympy_symbolic = df_dx_sympy_symbolic.diff(xs)
    d2f_dx_dy_sympy_symbolic = df_dx_sympy_symbolic.diff(ys)
    d2f_dy_dy_sympy_symbolic = df_dy_sympy_symbolic.diff(ys)

    values = {xs: sy.E, ys: sy.pi}

    f_sympy = f_sympy_symbolic.evalf(subs=values)
    df_dx_sympy = df_dx_sympy_symbolic.evalf(subs=values)
    df_dy_sympy = df_dy_sympy_symbolic.evalf(subs=values)
    d2f_dx_dx_sympy = d2f_dx_dx_sympy_symbolic.evalf(subs=values)
    d2f_dx_dy_sympy = d2f_dx_dy_sympy_symbolic.evalf(subs=values)
    d2f_dy_dy_sympy = d2f_dy_dy_sympy_symbolic.evalf(subs=values)

    print('f_sympy_symbolic')
    print(f_sympy_symbolic)
    print(f_sympy)

    print('df_dx_sympy_symbolic')
    print(df_dx_sympy_symbolic)
    print(df_dx_sympy)

    print('df_dy_sympy_symbolic')
    print(df_dy_sympy_symbolic)
    print(df_dy_sympy)

    print('d2f_dx_dx_sympy_symbolic')
    print(d2f_dx_dx_sympy_symbolic)
    print(d2f_dx_dx_sympy)

    print('d2f_dx_dy_sympy_symbolic')
    print(d2f_dx_dy_sympy_symbolic)
    print(d2f_dx_dy_sympy)

    print('d2f_dy_dy_sympy_symbolic')
    print(d2f_dy_dy_sympy_symbolic)
    print(d2f_dy_dy_sympy)

    f0 = f_2d(x, y)
    fd0: Dual = f_2d(Dual._lift(x), Dual._lift(y))
    fd1: Dual = f_2d(Dual(x, 1), Dual(y, 0))
    fd2: Dual = f_2d(Dual(x, 0), Dual(y, 1))
    print(f0)
    print(fd0)
    print(fd1)
    print(fd2)

    df_dx = fd1.second
    df_dy = fd2.second

    decimal = 13

    np.testing.assert_almost_equal(f0, f_sympy, decimal=decimal)
    np.testing.assert_almost_equal(fd0.first, f_sympy, decimal=decimal)
    np.testing.assert_almost_equal(fd0.second, 0.0, decimal=decimal)
    np.testing.assert_almost_equal(fd1.first, f_sympy, decimal=decimal)
    np.testing.assert_almost_equal(df_dx, df_dx_sympy, decimal=decimal)
    np.testing.assert_almost_equal(fd2.first, f_sympy, decimal=decimal)
    np.testing.assert_almost_equal(df_dy, df_dy_sympy, decimal=decimal)


if __name__ == "__main__":
    test_f_mul()
    test_f_1d()
    test_f_2d()
