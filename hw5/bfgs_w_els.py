from __future__ import division

import pandas as pd
import numpy as np
from numpy import dot
from functools import partial
from scipy.linalg import inv


def bfgs_els_update(x, B, c, H, dfp):
    p = dot(inv(B), -c - dot(H, x))
    a = - 1 / (dot(dot(p.T, H), p)) * dot((dot(H, x) + c).T, p)
    s = a * p
    x_next = x + s
    y = dot(H, s)
    B_next = B - 1 / (dot(dot(s.T, B), s)) * (
        dot(dot(dot(B, s), s.T), B.T)) + 1 / (dot(y.T, s)) * dot(y, y.T)
    if dfp:
        sbs = dot(dot(s.T, B), s)
        w = (1 / dot(y.T, s) * y) - (1 / sbs) * dot(B, s)
        B_next = B_next + sbs * dot(w, w.T)
    return (x_next, B_next, p, a, s, y)


def bfgs_gen(x, B, c, H, q, n=5, dfp=False):
    for i in range(n):
        xc = x
        Bc = B
        x, B, p, a, s, y = bfgs_els_update(x, B, c, H, dfp)
        yield ({'x': xc, 'B': Bc, 'f': q(x), 'p': p, 'a': a, 's': s, 'y': y})


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 80)

    def q(x):
        return (x[0] - 3./4. * x[1] + 4./9. * x[0] ** 2 -
                2 * x[0] * x[1] + 3 * x[1] * 2)

    def g(x):
        return np.array([1 + 8 / 9 * x[0] - 2 * x[1],
                        -.75 + 6 * x[1] - 2 * x[1]]).reshape(2, 1)

    fn = partial(np.round, decimals=4)
    x = np.array([-1, 4]).reshape(2, 1)
    B = np.array([2, 1, 1, 3]).reshape(2, 2)
    H = np.array([8/9, -2, -2, 6]).reshape(2, 2)
    c = np.array([1, -.75]).reshape(2, 1)

    p = dot(inv(B), -c - dot(H, x))
    a = - 1 / (dot(dot(p.T, H), p)) * dot((dot(H, x) + c).T, p)
    s = a * p
    y = dot(H, s)
    x1 = x + s
    B1 = B - 1 / (dot(dot(s.T, B), s)) * (
        dot(dot(dot(B, s), s.T), B.T)) + 1 / (dot(y.T, s)) * dot(y, y.T)

    k1 = []
    gen_bfgs = bfgs_gen(x, B, c, H, q)
    df = pd.DataFrame([ar for ar in gen_bfgs])
    df = df.applymap(fn)
    with open('hw5_1_results.txt', 'w') as f:
        f.write(df.to_latex())

    with open('hw5_1_results_text.txt', 'w') as f:
        f.write('### With BFGS update ###\n')
        f.write(df.to_string())

    gen_dfp = bfgs_gen(x, B, c, H, q, dfp=True)
    df2 = pd.DataFrame([ar for ar in gen_dfp])
    df2 = df2.applymap(fn)
    with open('hw5_1_results.txt', 'a') as f:
        f.write('\n\nWith DFP\n\n')
        f.write(df2.to_latex())

    with open('hw5_1_results_text.txt', 'a') as f:
        f.write('\n\n### With DFP update ###\n\n')
        f.write(df2.to_string())
        f.write('\n\n### The Difference###\n\n')
        f.write((df - df2).to_string())
