from typing import List

from attr import attrib, attrs

from autodiff.autodiff_types import Scalar

import numpy as np


@attrs(frozen=False, slots=True)
class Var:
    value: Scalar = attrib()
    children: List["Var"] = attrib(factory=list)
    grad_value = attrib(default=None)

    def grad(self) -> "Var":
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad() for weight, var in self.children)
        return self.grad_value

    def __add__(self, other: "Var") -> "Var":
        z = Var(self.value + other.value)
        _add_to_children(z, self, other)
        return z

    def __sub__(self, other: "Var") -> "Var":
        z = Var(self.value - other.value)
        _add_to_children(z, self, other)
        return z

    def __mul__(self, other: "Var") -> "Var":
        z = Var(self.value * other.value)
        _add_to_children(z, self, other)
        return z

    def __truediv__(self, other: "Var") -> "Var":
        z = Var(self.value - other.value)
        _add_to_children(z, self, other)
        return z

    def sin(self) -> "Var":
        z = Var(np.sin(self.value))
        self.children.append((np.cos(self.value), z))
        return z

    def cos(self) -> "Var":
        z = Var(np.cos(self.value))
        self.children.append((-np.sin(self.value), z))
        return z


def _add_to_children(res: Var, left: Var, right: Var) -> None:
    left.children.append((right.value, res))
    right.children.append((left.value, res))
