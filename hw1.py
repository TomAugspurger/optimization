from __future__ import division

import numpy as np
from numpy import linalg

# 3.1
A = np.array([[-2, 0], [0, 2]])
r, v = linalg.eig(A)
print(r)
try:
    print(linalg.cholesky(A))
except linalg.linalg.LinAlgError, e:
    print('e')

# 3.2
print('\nPart 2')
A = np.array([[6, 0], [0, -16]])
B = np.array([[6, 0], [0, 24 * (np.sqrt(2)) ** 2 - 16]])
r, v = linalg.eig(A)
r2, v2 = linalg.eig(B)
print(r)

try:
    print(linalg.cholesky(A))
except linalg.linalg.LinAlgError, e:
    print(e)

print("\nSecond Solution")
print(r2)
try:
    print(linalg.cholesky(B))
except linalg.linalg.LinAlgError, e:
    print('e')

# 3.3
print('\nPart 3')
A = np.array([[24, -12], [-24, -12]])
r, v = linalg.eig(A)
print(r)
try:
    print(linalg.cholesky(A))
except linalg.linalg.LinAlgError, e:
    print(e)


def m(x):
    return np.array([[-2 * np.cos(x[0]) * x[1], -2 * np.sin(x[0])],
        [-2 * np.sin(x[0]), 2]])
