from __future__ import division

import pandas as pd
import numpy as np
from numpy import dot
from functools import partial

def bfgs_els_update(x, B, c, H, dfp):
    p = -c - dot(H, x)
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


def bfgs_gen(x, B, c, H, q, n=4, dfp=False):
    for i in range(n):
        x, B, p, a, s, y = bfgs_els_update(x, B, c, H, dfp)
        yield ({'x': x, 'B': B, 'f': q(x), 'p': p, 'a': a, 's': s, 'y': y})


def dfp_full_update(x, B, c, H):
    p = -c - dot(H, x)
    a = - 1 / (dot(dot(p.T, H), p)) * dot((dot(H, x) + c).T, p)
    s = a * p
    x_next = x + s
    y = dot(H, s)
    sBs = dot(dot(s.T, B), s)
    BssB = dot(dot(dot(B, s), s.T), B.T)
    ys = dot(y.T, s)
    yy = dot(y, y.T)
    B_next = B - (1 / sBs) * BssB + (1 / ys) * yy + sBs * dot((1 / ys) * y - (1 / sBs) * dot(B, s), ((1 / ys) * y - (1 / sBs) * dot(B, s)).T)
    return (x_next, B_next, p, a, s, y)


def dfp_full_gen(x, B, c, H, q, n=4):
    for i in range(n):
        x, B, p, a, s, y = dfp_full_update(x, B, c, H)
        yield ({'x': x, 'B': B, 'f': q(x), 'p': p, 'a': a, 's': s, 'y': y})

fn = partial(np.round, decimals=4)
gen = dfp_full_gen(x, B, c, H, q)

df = pd.DataFrame([ar for ar in gen])
df = df.applymap(fn)
