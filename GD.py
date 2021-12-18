import math
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sympy as sym
import functions as fnc
import matplotlib.cm as cm




# Question 1
""""
x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = meshgrid(x, y)  #  grid met punten
Z = fnc.z_func(X, Y)  #Evaluatie van z op de grid

# 2d plot
fnc.plot2d(X,Y,Z)
# 3d plot
fnc.plot3d(X,Y,Z)

"""
# Question 3 - Local search
"""
x = 0
y = 0
num_iter = 15 # number of iterations
alpha = 0.9 # tuning parameter

x_coord, y_coord, x_diff, y_diff = fnc.GD(x,y, num_iter, alpha)
# plot verloop positie
# x coordinaat
plt.plot(x_coord)
plt.xlabel("Iteration")
plt.ylabel("x-value of C_{n}")
plt.show()
# op 2d plot
Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')
plt.plot(x_coord[0], y_coord[-0], 'or', markersize=5)
plt.plot(x_coord, y_coord, '-.r', markersize=10, linewidth=1, label="iteration steps")
plt.plot(x_coord[-1], y_coord[-1], 'oy', markersize=5)
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

zs = []
for iter in range(len(x_coord)):
    Z2 = fnc.z_func(x_coord[iter], y_coord[iter])  #Evaluatie van z op de grid
    zs.append(Z2)
print(zs)
"""
# Question 4 and 5
"""
x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = meshgrid(x, y)  #  grid met punten - origineel
Z = fnc.z_func(X, Y)  #Evaluatie van z op de grid
x2 = np.arange(-4.0, 5.0, 2)
y2 = np.arange(-4.0, 5.0, 2)
X2, Y2 = meshgrid(x2, y2)  # begin locaties goalkeeper

Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')
plt.plot(X2, Y2, 'or')
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = meshgrid(x, y)  #  grid met punten - origineel
Z = fnc.z_func(X, Y)  #Evaluatie van z op de grid
Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')

x2 = np.arange(-4.0, 5.0, 2)
y2 = np.arange(-4.0, 5.0, 2)
colors = cm.rainbow(np.linspace(0, 1, len(x2)))
num_iter = 2000
alpha = -0.9
for x in x2:
    for y, c in zip(y2, colors):
        x_coord, y_coord, x_diff, y_diff = fnc.GD(x, y, num_iter, alpha)
        z = fnc.z_func(x_coord[-1], y_coord[-1])
        plt.plot(x_coord[0], y_coord[-0], 'or', markersize = 5)
        plt.plot(x_coord, y_coord, '-.r', markersize = 10, linewidth=1)
        plt.plot(x_coord[-1], y_coord[-1], 'oy', markersize = 5)
        print(x_coord[-1])
        print(y_coord[-1])
        print(z)
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
# Question 6
x = 0
y = 0
num_iter = 15 # number of iterations
"""
alpha = np.arange(0.1, 2.1, 0.2)  # tuning parameter
for alp in alpha:
    x_coord, y_coord, x_diff, y_diff = fnc.GD(x,y, num_iter, alp)
    # plot verloop positie
    # x coordinaat
    plt.plot(x_coord)
    plt.title("Alpha = "+ str(round(alp,1)))
    plt.xlabel("Iteration [i]")
    plt.ylabel("x-value of $C_{i}$")
    plt.show()
"""

alpha = 4
factor = 0.8

x_coord, y_coord, x_diff, y_diff = fnc.GD_alpha(x, y, num_iter, alpha, factor)
# plot verloop positie
# x coordinaat
plt.plot(x_coord)
plt.title("Alpha = " + str(alpha) + ", factor = " + str(factor))
plt.xlabel("Iteration [n]")
plt.ylabel("x-value of $C_{n}$")
plt.show()

