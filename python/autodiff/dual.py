"""An implementation of single-variable forward-mode automatic differentiation
based on dual numbers represented as two components.
"""

import numpy as np
from attr import attrib, attrs

from autodiff.autodiff_types import DNumber, Number, Scalar


@attrs(frozen=True, slots=True)
class Dual:
    """An implementation of dual numbers (https://en.wikipedia.org/wiki/Dual_number).

    `first` contains the usual number, and `second` contains the coefficient
    of the nilpotent epsilon term, which is the first derivative of `first`
    automatically calculated. This scheme gives forward-mode automatic
    differentiation.
    """

    first: Number = attrib()
    second: Number = attrib()

    @staticmethod
    def lift(primitive: Number) -> "Dual":
        return Dual(primitive, 0)

    def __add__(self, other: DNumber) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.first + other.first, self.second + other.second)
        return self + Dual.lift(other)

    def __radd__(self, other: DNumber) -> "Dual":
        return self + other

    def __sub__(self, other: DNumber) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.first - other.first, self.second - other.second)
        return self - Dual.lift(other)

    def __rsub__(self, other: DNumber) -> "Dual":
        return self - other

    def __mul__(self, other: DNumber) -> "Dual":
        if isinstance(other, Dual):
            first = self.first * other.first
            second = self.second * other.first + self.first * other.second
            return Dual(first, second)
        return self * Dual.lift(other)

    def __rmul__(self, other: DNumber) -> "Dual":
        return self * other

    def __truediv__(self, other: DNumber) -> "Dual":
        if isinstance(other, Dual):
            if other.first == 0:
                raise Exception
            first = self.first / other.first
            second = ((self.second * other.first) - (self.first * other.second)) / (
                other.first ** 2
            )
            return Dual(first, second)
        return self / Dual.lift(other)

    def sin(self) -> "Dual":
        return Dual(np.sin(self.first), self.second * np.cos(self.first))

    def cos(self) -> "Dual":
        return Dual(np.cos(self.first), -self.second * np.sin(self.first))

    def exp(self) -> "Dual":
        return Dual(np.exp(self.first), self.second * np.exp(self.first))

    def log(self) -> "Dual":
        if self.first <= 0:
            raise Exception(
                f"The logarithm of a negative number is undefined: {self.first}"
            )
        return Dual(np.log(self.first), self.second / self.first)

    def __pow__(self, k: Number) -> "Dual":
        if self.first == 0:
            raise Exception
        first = self.first ** k
        second = k * (self.first ** (k - 1)) * self.second
        return Dual(first, second)

    def __abs__(self) -> "Dual":
        if self.first == 0:
            raise Exception
        return Dual(abs(self.first), self.second * np.sign(self.first))


def sin(n: DNumber) -> DNumber:
    if isinstance(n, Dual):
        return n.sin()
    return np.sin(n)


def cos(n: DNumber) -> DNumber:
    if isinstance(n, Dual):
        return n.cos()
    return np.cos(n)


def exp(n: DNumber) -> DNumber:
    if isinstance(n, Dual):
        return n.exp()
    return np.exp(n)


def log(n: DNumber) -> DNumber:
    if isinstance(n, Dual):
        return n.log()
    return np.log(n)
