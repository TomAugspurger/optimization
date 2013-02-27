from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy import dot
from scipy.linalg import inv

# TODO: Add initial x conditions to DataFrame.


def update(C, F, x, B):
    """
    The updating logic behind Broyden's method.
    """
    f_at_x = F(x)
    p = - dot(C, f_at_x)
    x = x + p
    new_f_at_x = F(x)
    y = new_f_at_x - f_at_x
    C_dot_y = dot(C, y)
    p_dot_C = dot(p.T, C)
    C = C + dot((p - C_dot_y), p_dot_C) / (dot(p.T, C_dot_y))
    f_at_x = new_f_at_x
    B = inv(C)
    return (C, F, x, B, f_at_x)


def broyden_gen(F, x, B, C, n=5):
    """
    Generator implementing Broyden's update.
    """
    for i in range(n):
        C, F, x, B, f_at_x = update(C, F, x, B)
        print('The estimate at {} is {}'.format(i, x))
        yield (x, B, f_at_x, C)


if __name__ == "__main__":
    # <nbformat>2</nbformat>
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
    a = broyden_gen(F, x0, B, C)
    print(a)

    # <markdowncell>
    # Showing that $(x_k)_1 + (x_k)_2 - 3 = 0 \forall k \geq 1$
        
    # <codecell>
    df = pd.DataFrame([x for x in a], columns=['x', 'B', 'f_at_x', 'C'])
    print(df['x'].apply(lambda x: x[0] + x[1] - 3))  # Should be zeros.
    
    # <markdowncell>
    # Showing that $(B_{k+1} - B_k)[1, 1]^T = 0 \forall k \geq 1$.
    # so that $(B_k)_{11} + (B_k)_{12} = 0 \forall k \geq 1$.
    
