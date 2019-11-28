# https://stackoverflow.com/questions/39558515/how-to-get-the-gradient-and-hessian-sympy

from sympy import Function, Matrix, Symbol, simplify, symbols
from sympy.tensor.array import derive_by_array

eta, xi, sigma = symbols("eta xi sigma")

x = Matrix([[xi], [eta]])

h = [Function("h_" + str(i + 1))(x[0], x[1]) for i in range(3)]
z = [Symbol("z_" + str(i + 1)) for i in range(3)]

lamb = 0
for i in range(3):
    lamb += 1 / (2 * sigma ** 2) * (z[i] - h[i]) ** 2
lamb = simplify(lamb)

gradient = derive_by_array(lamb, (eta, xi))
print(type(gradient))
print(gradient)

hessian = derive_by_array(gradient, (eta, xi))
print(type(hessian))
print(hessian)
