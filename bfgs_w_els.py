from __future__ import division

import pandas as pd
import numpy as np
from numpy import dot


def bfgs_els_update(x, B, c, H):
    p = -c - dot(H, x)
    a = - 1 / (dot(dot(p.T, H), p)) * dot((dot(H, x) + c).T, p)
    s = a * p
    x_next = x + s
    y = dot(H, s)
    B_next = B - 1 / (dot(dot(s.T, B), s)) * (
        dot(dot(dot(B, s), s.T), B.T)) + 1 / (dot(y.T, s)) * dot(y, y.T)
    return (x_next, B_next, p, a, s, y)


def bfgs_gen(x, B, c, H, q, n=4):
    for i in range(n):
        x, B, p, a, s, y = bfgs_els_update(x, B, c, H)
        yield ({'x': x, 'B': B, 'f': q(x), 'p': p, 'a': a, 's': s, 'y': y})

if __name__ == '__main__':
    def q(x):
        return (x[0] - 3./4. * x[1] + 4./9. * x[0] ** 2 -
                2 * x[0] * x[1] + 3 * x[1] * 2)

    def g(x):
        return np.array([1 + 8 / 9 * x[0] - 2 * x[1],
                        -.75 + 6 * x[1] - 2 * x[1]]).reshape(2, 1)

    x = np.array([-1, 4]).reshape(2, 1)
    B = np.array([2, 1, 1, 3]).reshape(2, 2)
    H = np.array([8/9, -2, -2, 6]).reshape(2, 2)
    c = np.array([1, -.75]).reshape(2, 1)

    gen = bfgs_gen(x, B, c, H, q)
    df = pd.DataFrame([ar for ar in gen])
