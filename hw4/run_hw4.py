from __future__ import division
from __future__ import print_function

import itertools as it

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import inv

from broyden import broyden_gen
from newton import gen_newton

if __name__ == "__main__":
    # Number 1
    # <codecell>
    def F(x):
        x.reshape(2, 1)
        return np.array([
                        [x[0] ** 2 + x[1] ** 2 - 9],
                        [x[0] + x[1] - 3]]).reshape(2, 1)
    # <codecell>
    x0 = np.array([5, 1]).reshape(2, 1)
    # f_at_x = F(x0)
    B = np.array([[10, 2], [1, 1]])
    C = inv(B)
    # <markdowncell>
    # Showing that for $k \geq 0, (B_k)_{21} = 1, (B_k)_{22} = 1$

    # <codecell>
    a = broyden_gen(F, x0, B, C, F(x0), n=2)


    # <markdowncell>
    # Showing that $(x_k)_1 + (x_k)_2 - 3 = 0 \forall k \geq 1$

    # <codecell>
    df = pd.DataFrame([x for x in a], columns=['x', 'B', 'f_at_x', 'C'])
    with open('hw4_results.txt', 'w') as f:
        f.write('\nNumber 1. Broyden\'s Method\n')
        f.write('\n\n'.join([df['x'].to_string(name=True), df['B'].to_string(name=True),
                    df['C'].to_string(name=True),
                    df['f_at_x'].to_string(name=True)]))
    # print(df['x'].apply(lambda x: x[0] + x[1] - 3))  # Should be zeros.

    # <markdowncell>
    # Showing that $(B_{k+1} - B_k)[1, 1]^T = 0 \forall k \geq 1$.
    # so that $(B_k)_{11} + (B_k)_{12} = 0 \forall k \geq 1$.

    # Number 2.
    def F(x):
        return np.array(
            [np.sin(x[0] * np.exp(3 * x[1]) - 1),
                x[0] ** 3 * x[1] + x[0] ** 3 - 7 * x[1] - 1])

    def g(x):
        return np.array(
            [np.exp(3 * x[1]) * np.cos(1 - np.exp(3 * x[1]) * x[0]),
            3 * np.exp(3 * x[1]) * x[0] * np.cos(1 - np.exp(3 * x[1]) * x[0]),
            3 * x[0] ** 2 * x[1] + 3 * x[0], x[0] ** 3 - 7]
            ).reshape(2, 2)
    x0 = np.array([[1.3], [-0.15]])
    x_f = np.array([[1], [0]])
    gen = gen_newton(x0, F, g, 1000)

    # gen will yield x, f_x; only need x (i.e. x[0]) for takewhile
    gen2 = it.takewhile(lambda x: norm(x_f - x[0], 2) > 10e-14, gen)
    df = pd.DataFrame([i for i in gen2], columns=['x', 'f_x'])
    df['difference'] = df['x'].apply(lambda x: x - x_f)
    df['error'] = df['difference'].apply(lambda x: norm(x, ord=2))
    with open('hw4_results.txt', 'a') as fout:
        fout.write('\n' + 79*'#')
        fout.write('\n\nNumber 2 Part a: Newton\'s Method\n\n')
        fout.write('Estimate of x\n')
        fout.write(df['x'].to_string())
        fout.write('\n\nActual Error\n')
        fout.write(df['error'].to_string())
        fout.write('\n' + 79*'#')

    # Number 2 part b: same function via Broyden's method.
    B = g(x0)
    C = inv(B)
    a = broyden_gen(F, x0, B, C, F(x0), n=100)
    a2 = it.takewhile(lambda x: norm(x_f - x[0], 2) > 10e-14, a)
    df = pd.DataFrame([x for x in a2], columns=['x', 'B', 'f_at_x', 'C'])
    df['difference'] = df['x'].apply(lambda x: x - x_f)
    df['error'] = df['difference'].apply(lambda x: norm(x, ord=2))

    df = df[['x', 'error']]
    with open('hw4_results.txt', 'a') as fout:
        fout.write('\n\nNumber 2 Part b: Broyden\'s Method\n\n')
        fout.write('\nEstimate of x\n' + df['x'].to_string() + '\n\n')
        fout.write('Actual Error\n' + df['error'].to_string())
