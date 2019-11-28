"""An implementation of forward-mode automatic differentiation based on dual
numbers represented as matrices."""

from abc import ABC
from typing import Optional, Union

import numpy as np
from attr import attrib, attrs
from attr.validators import instance_of

from autodiff.autodiff_types import Scalar


@attrs(frozen=True, slots=True)
class DualMatND(ABC):
    mat: np.ndarray = attrib(validator=instance_of(np.ndarray))

    @staticmethod
    def from_vals(vals) -> "DualMatND":
        dim = len(vals)
        assert dim >= 2
        _type = type(vals[0])
        mat = np.zeros((dim, dim), dtype=_type)
        for i, val in enumerate(vals):
            # This does a bunch of extra work, writing pointless values...
            mat[np.triu_indices_from(mat, i)] = val
        return DualMatND(mat)

    @staticmethod
    def lift(primitive: Scalar, order: Optional[int]) -> "DualMatND":
        if order:
            assert order >= 1
        else:
            order = 2
        vals = [0 for _ in range(order)]
        vals[0] = primitive
        return DualMatND.from_vals(vals)

    @property
    def first(self) -> Scalar:
        return self.mat[0, 0]

    @property
    def second(self) -> Scalar:
        return self.mat[0, 1]

    def __add__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        if isinstance(other, DualMatND):
            return DualMatND(self.mat + other.mat)
        return self + DualMatND.lift(other)

    def __radd__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        return self + other

    def __sub__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        if isinstance(other, DualMatND):
            return DualMatND(self.mat - other.mat)
        return self - DualMatND.lift(other)

    def __rsub__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        return self - other

    def __mul__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        if isinstance(other, DualMatND):
            return DualMatND(self.mat @ other.mat)
        return self * DualMatND.lift(other)

    def __rmul__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        return self * other

    def __truediv__(self, other: Union["DualMatND", Scalar]) -> "DualMatND":
        if isinstance(other, DualMatND):
            return DualMatND(self.mat @ np.linalg.inv(other.mat))
        return self / DualMatND.lift(other)


@attrs(frozen=True, slots=True)
class DualMat2D(DualMatND):
    @staticmethod
    def from_vals(first: Scalar, second: Scalar) -> "DualMat2D":
        _type = type(first)
        dim = 2
        mat = np.zeros((dim, dim), dtype=_type)
        for i in range(dim):
            mat[i, i] = first
        mat[0, 1] = second
        return DualMat2D(mat)

    @staticmethod
    def lift(primitive: Scalar) -> "DualMat2D":
        return DualMat2D.from_vals(primitive, 0)
