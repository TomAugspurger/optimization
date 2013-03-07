from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy import dot
from scipy.linalg import inv
from scipy import optimize as opt

from broyden import broyden_gen, update

def newton_update(F, x, g):
    """
    """
    f_x = F(x)
    x_next = x - dot(inv(g(x)), f_x)
    return (x_next, f_x)


def gen_newton(x, F, g, n):
    """A generator version.

    Call like:
        gen = enumerate(gen_newton(x0, f, g, 10))
        [x for x in gen]
    """
    # if simple:  Broken
    #     g = lambda x: g(self.x0)  # Redefine g to always return A.

    for i in range(n):
        x, f_x = newton_update(F, x, g)
        yield x, f_x

if __name__ == '__main__':
    import itertools as it
    from numpy.linalg import norm

    def F(x):
        return np.array(
            [np.sin(x[0] * np.exp(3 * x[1]) - 1),
            x[0] ** 3 * x[1] + x[0] ** 3 - 7 * x[1] - 1])

    def g(x):
        return np.array(
            [np.exp(3 * x[1]) * np.cos(1 - np.exp(3 * x[1]) * x[0]),
            3 * np.exp(3 * x[1]) * x[0] * np.cos(1 - np.exp(3 * x[1]) * x[0]),
            3 * x[0] ** 2 * x[1] + 3 * x[0], x[0] ** 3 - 7
            ]).reshape(2, 2)
    x0 = np.array([[1.3], [-0.15]])
    x_f = np.array([[1], [0]])
    gen = gen_newton(x0, F, g, 1000)

    # gen will yield x, f_x; only need x (i.e. x[0]) for takewhile
    gen2 = it.takewhile(lambda x: norm(x_f - x[0], 2) > 10e-14, gen)
    df = pd.DataFrame([i for i in gen2], columns=['x', 'f_x'])
    df['error'] = df['x'].apply(lambda x: x - x_f)
    with open('hw4_results.txt', 'a') as fout:
        fout.write('\n\nNumber 2 Part a:\n\n')
        fout.write(df.to_string())

    # Number 2 part b: same function via Broyden's method.
    B = g(x0)
    C = inv(B)
    a = broyden_gen(F, x0, B, C, F(x0), n=100)
    a2 = it.takewhile(lambda x: norm(x_f - x[0], 2) > 10e-14, a)
    df = pd.DataFrame([x for x in a2], columns=['x', 'B', 'f_at_x', 'C'])
    df['error'] = df['x'].apply(lambda x: x - x_f)
    df = df[['x', 'error']]
    with open('hw4_results.txt', 'a') as fout:
        fout.write('\n\nNumber 2 Part b:\n\n')
        fout.write(df.to_string())
