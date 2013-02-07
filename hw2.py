from __future__ import division

import numpy as np
from scipy import optimize as opt


class Opt(object):
    """Various optimization routines.
    """
    def __init__(self, f, a=None, b=None, g=None, x0=None, tol=1e-10,
        verbose=False, max_iter=1000):
        """
        f: function to be minimized
        g: gradient of f.
        a: left endpoint
        b: right endpoint
        tol: maximum change between two iterates.
        verbose: True returns more detail.
        """
        self.f = f
        self.g = g
        self.a = a
        self.b = b
        self.x0 = x0
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter

    def bisection(self):
        """Finds the minimum via the bisection method.
        """
        a = self.a
        b = self.b
        f = self.f

        if np.dot(f(a), f(b)) >= 0:
            raise ValueError("f(a) and f(b) Must have different signs.")

        iterates = []
        c_next = (a + b) / 2
        e = 1

        max_k = int(np.ceil(np.log(np.abs(b - a) / self.tol) / np.log(2) - 1))
        for i in range(max_k):
            c = c_next
            if np.sign(f(c)) == np.sign(f(a)):
                a = c
            else:
                b = c
            c_next = (a + b) / 2
            e = np.abs(c - c_next)
            if self.verbose:
                iterates.append((c, e))
        return (c, e, max_k, iterates)

    ## Number 2

    def newton(self, simple=False):
        f = self.f
        g = self.g
        if g is None:
            raise ValueError('Must provided a gradient.')
        A = g(self.x0)  # Approximation to the Hessian.
        if simple:
            g = lambda x: A  # Redefine g to always return A.
        e = 1
        x = x0
        k = 0
        out = []

        while e > self.tol and k <= self.max_iter:
            x_next = x - f(x) / g(x)
            e = np.abs(x - x_next)
            k += 1
            if self.verbose:
                out.append((k, x_next, e))
            x = x_next
        return (x_next, e, k, out)

    def secant(self, x1=None):
        """
        Need either x1 or the gradient so that x1 may be computed.
        If x1 is given then the gradient will not be used.
        x_next = x_k+1; x = x_k, y = x_k-1.
        """
        f = self.f
        g = self.g
        if x1 == None:
            y = x0 - f(x0) / g(x0)  # First iteration from Newtons's Method.
        else:
            y = x1
        e = 1
        x = x0
        k = 0
        out = []

        while e > self.tol and k <= self.max_iter:
            A = (f(x) - f(y)) / (x - y)
            x_next = x - f(x) / A
            e = np.abs(x - x_next)
            k += 1
            if self.verbose:
                out.append((k, x_next, e))
            x, y = x_next, x  # I love Python.
        return (x_next, e, k, out)

    def fixed_point(self, T):
        """Use Contraction Mapping Theorem to find a zero for f.
        T: the operator on f, giving the next iterate.
        """
        self.f = f
        x = self.x0
        e = 1
        k = 0
        out = []

        while e > self.tol and k <= self.max_iter:
            x_next = T(x)
            e = np.abs(x - x_next)
            k += 1
            if self.verbose:
                out.append((k, x_next, e))
            x = x_next
        return (x_next, e, k, out)

    def golden_section(self):
        f = self.f
        a = self.a
        b = self.b
        tau = 2 / (1 + np.sqrt(5))
        k = 0
        full = []

        while k <= self.max_iter:
            a_next = a + (1 - tau) * (b - a)
            b_next = a + tau * (b - a)
            k += 1
            if f(a_next) < f(b_next):
                b = b_next
            else:
                a = a_next
            if self.verbose:
                full.append((k, (a_next, b_next)))
        return ((k, a_next, b_next, full))

if __name__ == '__main__':
    f = lambda x: np.sqrt(x) * np.exp(x) - 1
    g = lambda x: np.exp(x) * (.5 * x ** (-.5) + x ** (1.5))
    a, b = .1, 1
    x0 = .55

    m = Opt(f, a=a, b=b, g=g, x0=.55, verbose=True)
    zero, e, max_k, iterates = m.bisection()
    print('\n Number 1.')
    print('\n The necessary number of iterations is %i' % max_k)
    print('My solution: %f' % zero)
    print('Scipy solution: %f' % opt.bisect(f, a, b))

    print('\n Part 2: Simplified Newton\'s Method')

    A_0 = g(x0)
    zero, e, k, full = m.newton(simple=True)
    print(zero)

    print('\n Part c: Newton\'s Method')
    zero, e, k, full = m.newton()
    print(zero)

    print('\n Part d: Secant Method')
    zero, e, k, full = m.secant()
    print(zero)

    print('\n Part e: Fixed Point Method')
    T = lambda x: np.exp(-2 * x)
    zero, e, k, full = m.fixed_point(T)
    print(zero)

    #### Number 3
    print('\n Number 3')
    f = lambda x: 4 * x ** 5 - x ** 3 - x
    g = lambda x: 20 * x ** 4 - 3 * x ** 2 - 1
    x0 = .5

    n3 = Opt(f, g=g, x0=x0, verbose=True, max_iter=10)
    zero, e, k, full = n2.newton()
    print(full)
    print('Not Converging.  Cycling.')

    #### Number 4
    print('\n Number 4')
    f = lambda x: x ** 6 - (1.5) ** 6
    x0 = .25
    x1 = 5
    n4 = Opt(f, max_iter=6, verbose=True)
    zero, e, k, full = n4.secant(x1=x1)
    print(full)

    #### Number 5
    print('\n Number 5')
    f = lambda x: 2 * x ** (1.5) + 3 * np.exp(-x) + 5
    a0, b0 = 0.2, 0.6
    n5 = Opt(f, a=.2, b=.6, max_iter=3, verbose=True)
    zero, e, k, full = n5.golden_section()
    print(full)
