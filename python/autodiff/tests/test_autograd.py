import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad


def test_readme():
    def tanh(x):
        y = np.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

    grad_tanh = grad(tanh)
    assert grad_tanh(1.0) == 0.41997434161402603
    assert (tanh(1.0001) - tanh(0.9999)) / 0.0002 == 0.41997434264973155

    egrad_tanh = egrad(tanh)
    assert egrad_tanh(1.0) == 0.41997434161402603
