import attr
from typing import Union

import numpy as np


Scalar = Union[float, int]


@attr.s
class DualMat2D:
    first: Scalar = attr.ib()
    second: Scalar = attr.ib()
    mat: np.ndarray = attr.ib(init=False)
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
