from __future__ import division

import numpy as np
import pandas as pd
# from broyden import *
from scipy import optimize as opt


class Opt(object):
    """Various optimization routines.
    Parameters
    ----------

    * f: The function to be minimized. (Currently just R -> R.)
    * g: Gradient of the function (for Newton's method).
    * a: Left starting point for bisection method.
    * b: right starting point for bisection method.
    * x0: Starting guess for various methods.
    * tol: An (absolute) delta between iterations.
    * verbose: Bool.  How much output to return.
    * max_iter: in case things go wrong.

    Example:
        f = lambda x: np.sqrt(x) * np.exp(x) - 1
        g = lambda x: np.exp(x) * (.5 * x ** (-.5) + x ** (1.5))
        a, b = .1, 1
        x0 = .55

        m = Opt(f, a=a, b=b, g=g, x0=.55, verbose=True)
        zero, e, max_k, iterates = m.bisection()
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

        c_next = (a + b) / 2
        iterates = [(1, c_next, 1.)]  # 1.0 for default error.
        e = 1
        k = 1
        max_k = int(np.ceil(np.log(np.abs(b - a) / self.tol) / np.log(2) - 1))
        for i in range(max_k):
            c = c_next
            if np.sign(f(c)) == np.sign(f(a)):
                a = c
            else:
                b = c
            c_next = (a + b) / 2
            e = np.abs(c - c_next)
            k += 1
            if self.verbose:
                iterates.append((k, c, e))
        return (c, e, max_k, iterates)

    def newton(self, simple=False):
        f = self.f
        g = self.g
        if g is None:
            raise ValueError('Must provided a gradient.')
        A = g(self.x0)  # Approximation to the Hessian.
        if simple:
            g = lambda x: A  # Redefine g to always return A.
        e = 1
        x = self.x0
        k = 1
        out = [(0, self.x0, 1.0)]

        while e > self.tol and k <= self.max_iter:
            x_next = x - f(x) / g(x)
            e = np.abs(x - x_next)
            k += 1
            if self.verbose:
                out.append((k, x_next, e))
            x = x_next
        return (x_next, e, k, out)

    def gen_newton(self, x, f, g, n, simple=False):
        """A generator version.

        Call like:
            gen = enumerate(gen_newton(x0, f, g, 10))
            [x for x in gen]
        """
        # if simple:  Broken
        #     g = lambda x: g(self.x0)  # Redefine g to always return A.

        for i in range(n):
            x_next = x - f(x) / g(x)
            relative_delta = np.abs((x - x_next) / x_next)
            x = x_next
            yield x_next, relative_delta

    # def bhhh(self):
    #     """Quasi-Newton method where the Hessian is approsimated with
    #     (1/N) & sum(grad(ln(p)) * grad(ln(p)).T)
    #     """
    #     self.f = f
    #     x = self.x0
    #     e = 1
    #     k = 0
    #     N = len(self.x0)
    #     x = self.x0
    #     out = []

    #     while e > self.tol and k <= self.max_iter:
    #         A = (1 / N) * dot(g, g.T)
    #         x_next = x - f(x) / A
    #         delta = np.abs(x - x_next)

    def secant(self, x1=None):
        """
        Need either x1 or the gradient so that x1 may be computed.
        If x1 is given then the gradient will not be used.
        x_next = x_k+1; x = x_k, y = x_k-1.
        """
        f = self.f
        g = self.g
        x0 = self.x0
        if x1 == None:
            y = x0 - f(x0) / g(x0)  # First iteration from Newtons's Method.
        else:
            y = x1
        x = x0
        e = 1
        k = 1
        out = [(0, x0, 1.), (1, y, abs(x0 - y))]

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
        x = self.x0
        e = 1
        k = 1
        out = [(k, x, e)]

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
        full = [(k, (a, b))]

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

    def present(self, full, x):
        """Nicer formatting.  x_k is a sequence of guesses. x is the solution.
        full: tuple containing std output, (iteration, next_step, delta)
        x: the actual root.
        """
        df = pd.DataFrame(full, columns=['k', 'x_k', 'delta'])
        fig = (df['x_k'] - x).plot()
        fig.set_xlabel('Iteration')
        fig.set_ylabel('Error')
        df['error'] = df['x_k'] - x
        return (df[['x_k', 'error']], fig)
