from typing import Union

import numpy as np

Scalar = Union[float, int]
Number = Union[Scalar, np.ndarray]
DNumber = Union[Number, "Dual"]
