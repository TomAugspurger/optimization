import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


def f(x):
    return (3 - 3 * x[0] - 3 * x[1] + x[0] ** 2 + 2 * x[0] * x[1] +
        x[0] ** 2)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X = np.arange(-20, 20, .5)
Y = np.arange(-20, 20, .5)
X, Y = np.meshgrid(X, Y)
R = f([X, Y])

surf = ax.plot_surface(X, Y, R, cmap=cm.jet)
plt.draw()
plt.savefig('hw3_3_plot.png')

