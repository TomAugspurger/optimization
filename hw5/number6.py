import numpy as np
from scipy.linalg import inv
from numpy import dot
import pandas as pd

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
b = dot(A, x)


def cg_update(x, p, g, A):
    h = dot(A, p)
    a = -dot(g.T, p) / dot(h.T, p)
    x_next = x + a * p
    g_next = g + a * h
    b = dot(h.T, g_next) / dot(h.T, p)
    p_next = -g_next + b * p
    return (x_next, p_next, g_next)


def cg_gen(x, A, n=15):
    g = dot(A, x) - b
    p = -g
    for i in range(n):
        x, p, g = cg_update(x, p, g, A)
        yield x, p, g


def anorm(v, A):
    return np.sqrt(dot(dot(v.T, A), v))

if __name__ == '__main__':
    b = dot(A, x_star)
    gen = cg_gen(b, A)
    df = pd.DataFrame([x for x in gen], columns=['x', 'p', 'g'])
    denom = anorm(x_star, A)
    df['error'] = df.x.apply(lambda x: anorm(x, A) / denom)
    with open('hw5_part6.txt', 'w') as f:
        f.write('#' * 10 + 'Number 6' + 10 * '#' + '\n\n')
        f.write(df['error'].to_string(name=True))
