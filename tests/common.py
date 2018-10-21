import sympy

from dual import sin, log, exp


def f_1d(x):
    return (3.67 * (x ** 3)) - 2


def f_2d(x, y):
    return sin(log(7 * x) + exp(y)) + 9


def f_2d_sympy(x, y):
    return sympy.sin(sympy.log(7 * x) + sympy.exp(y)) + 9
