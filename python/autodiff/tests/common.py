from typing import Any

import sympy

from autodiff.autodiff_types import DNumber
from autodiff.dual import exp, log, sin


def f_1d(x) -> Any:
    return (3.67 * (x ** 3)) - 2


def f_2d(x, y) -> DNumber:
    return sin(log(7 * x) + exp(y)) + 9


def f_2d_sympy(x, y):
    return sympy.sin(sympy.log(7 * x) + sympy.exp(y)) + 9
