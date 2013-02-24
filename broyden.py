from __future__ import division

import numpy as np
from scipy import dot
from scipy.linalg import inv


# Initialization
def F(x):
    x.reshape(2, 1)
    return np.array([
                    [x[0] ** 2 + x[1] ** 2 - 9],
                    [x[0] + x[1] - 3]]).reshape(2, 1)


x = np.array([5, 1]).reshape(2, 1)
f_at_x = F(x)
B = np.array([[10, 2], [1, 1]])
C = inv(B)


def update(C, F, x, B, f_at_x):
    """
    The updating logic behind Broyden's method.
    """
    p = - dot(C, f_at_x)
    x = x + p
    new_f_at_x = F(x)
    y = new_f_at_x - f_at_x
    C_dot_y = dot(C, y)
    p_dot_C = dot(p.T, C)
    C = C + dot((p - C_dot_y), p_dot_C) / (dot(p.T, C_dot_y))
    f_at_x = new_f_at_x
    return (C, F, x, B, f_at_x)


def broyden_gen(C=C, F=F, x=x, B=B, f_at_x=f_at_x, n=5):
    """
    Generator implementing Broyden's update.
    """
    for i in range(n):
        C, F, x, B, f_at_x = update(C, F, x, B, f_at_x)
        yield x
