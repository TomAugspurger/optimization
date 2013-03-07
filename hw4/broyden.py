from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy import dot
from scipy.linalg import inv


def update(C, F, x, B, f_at_x):
    """
    The updating logic behind Broyden's method.
    """
    # f_at_x = F(x)
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


def broyden_gen(F, x, B, C, f_at_x, n=5):
    """
    Generator implementing Broyden's update.
    """
    for i in range(n):
        C, F, x, B, f_at_x = update(C, F, x, B, f_at_x)
        yield (x, B, f_at_x, C)
