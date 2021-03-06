from __future__ import division

import numpy as np
from scipy.linalg import inv
from numpy import dot
import pandas as pd
from functools import partial


def cg_update(x, p, g, A):
    """
    Update logic behind conjugate gradient method.
    See page 74 of typed notes.
    Parameters
    ----------
        x - vector for current position
        p - direction to step
        g - current gradient
        A - symmetric positive definite matrix
    Returns
    -------
        x_next - vector with the updated position
        p_next - vector with the updated direction
        g_next - vector with the updated gradient
    """
    h = dot(A, p)
    a = -dot(g.T, p) / dot(h.T, p)
    x_next = x + a * p
    g_next = g + a * h
    b = dot(h.T, g_next) / dot(h.T, p)
    p_next = -g_next + b * p
    return (x_next, p_next, g_next)


def cg_gen(x, A, n=15):
    """
    Generator to wrap the update logic for the conjugate gradient method.

    Parameters
    ----------
        x - vector with initial guess
        A - symmetric positive definite matrix
        n - Maximum number of iterations
    Returns
    -------
        generator whose .next() method contains x, p, and g.
    """
    g = dot(A, x) - b
    p = -g
    for i in range(n):
        x, p, g = cg_update(x, p, g, A)
        yield x, p, g


def anorm(v, A):
    return np.sqrt(dot(dot(v.T, A), v))

if __name__ == '__main__':
    A = np.zeros([15, 15])
    np.fill_diagonal(A, 2)

    for i, row in enumerate(A):
        if i == 0:
            row[i + 1] = -1
        elif i == len(A) - 1:
            row[i - 1] = -1
        else:
            row[i + 1] = -1
            row[i - 1] = -1

    x = np.arange(1, 16)
    x_star = ([1, -1] * 7 + [1]) * x
    denom = anorm(x_star, A)
    e_0 = anorm((x_star - x), A) / denom
    b = dot(A, x_star)
    gen = cg_gen(b, A)
    df = pd.DataFrame([v for v in gen], columns=['x', 'p', 'g'])
    df['error'] = df.x.apply(lambda x_k: x_star - x_k)
    df['norm_error'] = df.error.apply(lambda x: anorm(x, A) / denom)
    df.index = df.index + 1
    
    with open('hw5_part6.txt', 'w') as f:
        f.write('#' * 10 + 'Number 6' + 10 * '#' + '\n\n')
        f.write('0     ' + str(e_0) + '\n' + df['norm_error'].to_string(name=True))
