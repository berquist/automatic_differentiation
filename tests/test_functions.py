import numpy as np

from dual import Dual

from common import f_1d, f_2d


if __name__ == "__main__":
    x = 0.9
    d0 = Dual(x, 0)
    d1 = Dual(x, 1)
    print(f_1d(x))
    print(f_1d(d0))
    print(f_1d(d1))

    x = np.e
    y = np.pi
    f0 = f_2d(x, y)
    fd0 = f_2d(Dual._lift(x), Dual._lift(y))
    fd1 = f_2d(Dual(x, 1), Dual(y, 1))
    print(f0)
    print(fd0)
    print(fd1)
