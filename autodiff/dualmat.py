import attr
from typing import Optional, Union

import numpy as np


Scalar = Union[float, int]


@attr.s
class DualMat2D:
    first: Scalar = attr.ib()
    second: Scalar = attr.ib()
    mat: Optional[np.ndarray] = attr.ib()

    @mat.default
    def init_mat(self):
        _type = type(self.first)
        dim = 2
        mat = np.zeros((dim, dim), dtype=_type)
        for i in range(dim):
            mat[i, i] = self.first
        mat[0, 1] = self.second
        return mat

    @staticmethod
    def _lift(primitive: Scalar) -> 'DualMat2D':
        return DualMat2D(primitive, 0)

    def __add__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.first + other.first,
                             self.second + other.second,
                             self.mat + other.mat)
        else:
            return self + DualMat2D._lift(other)

    def __radd__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self + other

    def __sub__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.first - other.first,
                             self.second - other.second,
                             self.mat - other.mat)
        else:
            return self - DualMat2D._lift(other)

    def __rsub__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self - other

    def __mul__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            first = self.first * other.first
            second = self.second * other.first + self.first * other.second
            return DualMat2D(first, second, self.mat @ other.mat)
        else:
            return self * DualMat2D._lift(other)

    def __rmul__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self * other

    def __truediv__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            first = self.first / other.first
            second = ((self.second * other.first) - (self.first * other.second)) / (other.first ** 2)
            return DualMat2D(first, second, self.mat @ np.linalg.inv(other.mat))
        else:
            return self / DualMat2D._lift(other)
