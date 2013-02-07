from __future__ import division

import numpy as np
from scipy import optimize as opt

# Number 1.


def bisection(f, a, b, tol=1e-10, verbose=False):
    """Finds the minimum via the bisection method.
    a: left endpoint
    b: right endpoint
    g: function over [a, b]
    """
    if np.dot(f(a), f(b)) >= 0:
        raise ValueError("f(a) and f(b) Must have different signs.")
    if verbose:
        iterates = []
    else:
        iterates = None
    c_next = (a + b) / 2
    e = 1

    max_k = int(np.ceil(np.log(np.abs(b - a) / tol) / np.log(2) - 1))
    for i in range(max_k):
        c = c_next
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
        c_next = (a + b) / 2
        e = np.abs(c - c_next)
        if verbose:
            iterates.append((c, e))
    return (c, e, max_k, iterates)

## Number 2


def newton_simple(f, g, x0, tol=1e-10, max_iter=10000, verbose=False):
    """
    f: function
    g: gradient
    x0: inital guess
    """
    A = g(x0)  # Approximation to the Hessian.
    e = 1
    x = x0
    k = 0
    out = []
    while e > tol and k < max_iter:
        x_next = x - f(x) / A
        e = np.abs(x - x_next)
        k += 1
        if verbose:
            out.append((k, x_next, e))
        x = x_next
    return (x_next, e, k, out)

if __name__ == '__main__':
    f = lambda x: np.sqrt(x) * np.exp(x) - 1
    a, b = .1, 1
    x0 = .55
    # zero, e, max_k, iterates = bisection(f, a, b, tol=10e-7)
    # print('\n Number 1.')
    # print('\n The necessary number of iterations is %i' % max_k)
    # print('My solution: %f' % zero)
    # print('Scipy solution: %f' % opt.bisect(f, a, b))

    print('\n Number 2')
    g = lambda x: np.exp(x) * (.5 * x ** (-.5) + x ** (1.5))
    A_0 = g(x0)
    zero, e, k, full = newton_simple(f, g, A_0)
    print(zero)
