import math

type
  Dual[T: SomeNumber] = object
    first: T
    second: T

### Fundamental operations

func add(left, right: Dual): Dual =
  Dual(first: left.first + right.first,
       second: left.second + right.second)

func sub(left, right: Dual): Dual =
  Dual(first: left.first - right.first,
       second: left.second - right.second)

func mul(left, right: Dual): Dual =
  Dual(first: left.first * right.first,
       second: left.second * right.first + left.first * right.second)

func `div`[T: SomeInteger](left, right: Dual[T]): Dual[T] =
  Dual[T](first: left.first div right.first,
          second: ((left.second * right.first) - (left.first * right.second)) div (right.first ^ 2))

func `div`[T: SomeFloat](left, right: Dual[T]): Dual[T] =
  Dual[T](first: left.first / right.first,
          second: ((left.second * right.first) - (left.first * right.second)) / (right.first ^ 2))

func sin[T: SomeNumber](d: Dual[T]): Dual[float] =
  Dual[float](first: float(d.first).sin(),
              second: float(d.second) * float(d.first).cos())

func cos[T: SomeNumber](d: Dual[T]): Dual[float] =
  Dual[float](first: float(d.first).cos(),
              second: float(-d.second) * float(d.first).sin())

func exp[T](d: Dual[T]): Dual[float] =
  let v = float(d.first).exp()
  Dual[float](first: v,
              second: float(d.second) * v)

func ln[T: SomeNumber](d: Dual[T]): Dual[float] =
  Dual[float](first: float(d.first).ln(),
              second: d.second / d.first)

func pow[T: SomeFloat](d: Dual[T], k: T): Dual[T] =
  Dual[T](first: d.first.pow(k),
          second: k * (d.first.pow(k - 1.0)) * d.second)

func pow[T: SomeInteger](d: Dual[T], k: T): Dual[T] =
  Dual[T](first: d.first ^ k,
          second: k * (d.first^ (k - 1)) * d.second)

# TODO pow:
#  - first float, second int
#  - first int, second float

func abs[T](d: Dual[T]): Dual[T] =
  Dual[T](first: d.first.abs(),
          second: T(float(d.second) * float(d.first.sgn())))

### aliases

func `+`(left, right: Dual): Dual =
  left.add(right)

func `-`(left, right: Dual): Dual =
  left.sub(right)

func `*`(left, right: Dual): Dual =
  left.mul(right)

func `/`(left, right: Dual): Dual =
  left.div(right)

func `^`(d: Dual, k: SomeNumber): Dual =
  d.pow(k)


when isMainModule:
  let
    a = Dual[int](first: 2, second: 3)
    b = Dual[int](first: 3, second: 4)
  echo "add: ", a.add(b)
  echo "sub: ", a.sub(b)
  echo "mul: ", a.mul(b)
  echo "div: ", a.div(b)
  echo "sin: ", a.sin()
  echo "cos: ", a.cos()
  echo "exp: ", a.exp()
  echo "ln : ", a.ln()
  echo "pow: ", a.pow(2)
  echo "abs: ", a.abs()

  let
    c = Dual[float](first: 2.0, second: 3.0)
    d = Dual[float](first: 3.0, second: 4.0)
  echo "add: ", c.add(d)
  echo "sub: ", c.sub(d)
  echo "mul: ", c.mul(d)
  echo "div: ", c.div(d)
  echo "sin: ", c.sin()
  echo "cos: ", c.cos()
  echo "exp: ", c.exp()
  echo "ln : ", c.ln()
  echo "pow: ", c.pow(2.0)
  echo "abs: ", c.abs()
