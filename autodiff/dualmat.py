import attr
from typing import Optional, Union

import numpy as np


Scalar = Union[float, int]


@attr.s
class DualMat2D:
    mat: np.ndarray = attr.ib()

    @staticmethod
    def from_vals(first: Scalar, second: Scalar) -> 'DualMat2D':
        _type = type(first)
        dim = 2
        mat = np.zeros((dim, dim), dtype=_type)
        for i in range(dim):
            mat[i, i] = first
        mat[0, 1] = second
        return DualMat2D(mat)

    @staticmethod
    def lift(primitive: Scalar) -> 'DualMat2D':
        return DualMat2D.from_vals(primitive, 0)

    @property
    def first(self) -> Scalar:
        return self.mat[0, 0]

    @property
    def second(self) -> Scalar:
        return self.mat[0, 1]

    def __add__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.mat + other.mat)
        else:
            return self + DualMat2D.lift(other)

    def __radd__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self + other

    def __sub__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.mat - other.mat)
        else:
            return self - DualMat2D.lift(other)

    def __rsub__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self - other

    def __mul__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.mat @ other.mat)
        else:
            return self * DualMat2D.lift(other)

    def __rmul__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        return self * other

    def __truediv__(self, other: Union['DualMat2D', Scalar]) -> 'DualMat2D':
        if isinstance(other, DualMat2D):
            return DualMat2D(self.mat @ np.linalg.inv(other.mat))
        else:
            return self / DualMat2D.lift(other)
