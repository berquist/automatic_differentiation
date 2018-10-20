import attr
from typing import Union

import numpy as np


Number = Union[float, int, np.ndarray]


@attr.s
class Dual:
    first: Number = attr.ib()
    second: Number = attr.ib()

    @staticmethod
    def _lift(primitive: Number) -> 'Dual':
        return Dual(primitive, 0)

    def __add__(self, other: Union['Dual', Number]) -> 'Dual':
        if isinstance(other, Dual):
            return Dual(self.first + other.first, self.second + other.second)
        else:
            return self + Dual._lift(other)

    def __radd__(self, other: Union['Dual', Number]) -> 'Dual':
        return self + other

    def __sub__(self, other: Union['Dual', Number]) -> 'Dual':
        if isinstance(other, Dual):
            return Dual(self.first - other.first, self.second - other.second)
        else:
            return self - Dual._lift(other)

    def __rsub__(self, other: Union['Dual', Number]) -> 'Dual':
        return self - other

    def __mul__(self, other: Union['Dual', Number]) -> 'Dual':
        if isinstance(other, Dual):
            return Dual(self.first * other.first, self.second * other.second)
        else:
            return self * Dual._lift(other)

    def __rmul__(self, other: Union['Dual', Number]) -> 'Dual':
        return self * other

    def __truediv__(self, other: Union['Dual', Number]) -> 'Dual':
        if isinstance(other, Dual):
            if other.first == 0:
                raise Exception
            first = self.first / other.first
            second = ((self.second * other.first) - (self.first * other.second)) / (other.first ** 2)
            return Dual(first, second)
        else:
            return self / Dual._lift(other)

    def sin(self) -> 'Dual':
        return Dual(np.sin(self.first), self.second * np.cos(self.first))

    def cos(self) -> 'Dual':
        return Dual(np.cos(self.first), -self.second * np.sin(self.first))

    def exp(self) -> 'Dual':
        return Dual(np.exp(self.first), self.second * np.exp(self.first))

    def log(self) -> 'Dual':
        if self.first <= 0:
            raise Exception
        return Dual(np.log(self.first), self.second / self.first)

    def __pow__(self, k: Number) -> 'Dual':
        return Dual(self.first ** k, k * (self.first ** (k - 1)) * self.second)

    def __abs__(self) -> 'Dual':
        return Dual(abs(self.first), self.second * np.sign(self.first))


UNumber = Union[Number, Dual]


def sin(n: UNumber) -> UNumber:
    if isinstance(n, Dual):
        return n.sin()
    else:
        return np.sin(n)


def cos(n: UNumber) -> UNumber:
    if isinstance(n, Dual):
        return n.cos()
    else:
        return np.cos(n)


def exp(n: UNumber) -> UNumber:
    if isinstance(n, Dual):
        return n.exp()
    else:
        return np.exp(n)


def log(n: UNumber) -> UNumber:
    if isinstance(n, Dual):
        return n.log()
    else:
        return np.log(n)
