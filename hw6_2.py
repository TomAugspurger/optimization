from __future__ import division

import numpy as np
from numpy import sqrt, dot
from numpy.linalg import inv, norm

f0 = 0.1 * np.array([4 * sqrt(3) + 3, -4 * sqrt(3) + 3]).reshape(2, 1)
df0 = np.eye(2)
f1 = 0.2 * np.array([-3, 4]).reshape(2, 1)
df1 = np.array([1, 1, -1 / sqrt(3), 1 / sqrt(3)]).reshape(2, 2)

lhs = norm(dot(inv(df1), f0))
rhs = norm(dot(inv(df1), f1))

assert lhs < rhs
