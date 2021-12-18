import math
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sympy as sym
import functions as fnc
import matplotlib.cm as cm
import random


# Question 7
# Simulated Annealing

#Initialisatie

x_coord = [3.14]
y_coord = [3.14]
z = [fnc.z_func(x_coord[0],y_coord[0])]
sigma = 0.1
num_iter = 1000
T = 1
factor = 0.9

for i in range(num_iter):
    x_uit, y_uit, z_uit = fnc.func_SA(x_coord[i],y_coord[i],sigma, T, factor, i)
    x_coord.append(x_uit)
    y_coord.append(y_uit)
    z.append(z_uit)

plt.plot(z)
plt.xlabel("iteration (i)")
plt.ylabel("value of $z(c(x_{i}, y_{i}))$")
plt.show()
print(x_coord[-1])
print(y_coord[-1])
print(z[-1])

x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = meshgrid(x, y)  #  grid met punten - origineel
Z = fnc.z_func(X, Y)  #Evaluatie van z op de grid
Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')
plt.plot(x_coord, y_coord, '-.r', markersize = 10, linewidth=1)
plt.plot(x_coord[-1], y_coord[-1], 'oy', markersize = 5)
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


"""
# Question 8
x_coord = [-1]
y_coord = [-1]
z = [fnc.z_func(x_coord[0],y_coord[0])]
sigma = 0.3
num_iter = 100

T = 40
factor = 0.8

for i in range(num_iter):
    x_uit, y_uit, z_uit = fnc.func_SA_seed(x_coord[i],y_coord[i],sigma, T, factor, i)
    x_coord.append(x_uit)
    y_coord.append(y_uit)
    z.append(z_uit)

plt.plot(z)
plt.xlabel("iteration (i)")
plt.ylabel("value of $z(c(x_{i}, y_{i}))$")
plt.title("$T_0$ = " + str(T) + ", f = " + str(factor))
plt.show()
"""
"""
## Question 9
x_coord = [-4]
y_coord = [-4]
z = [fnc.z_func(x_coord[0],y_coord[0])]
sigma = 0.1
num_iter = 1000

T = 1
f_t = 0.99
f_gd = 0.99
alpha = -1

for i in range(num_iter):
    x_uit, y_uit, z_uit = fnc.func_SA_GD(x_coord[i],y_coord[i], sigma, T, f_t, i, f_gd, alpha)
    x_coord.append(x_uit)
    y_coord.append(y_uit)
    z.append(z_uit)

plt.plot(z)
plt.xlabel("iteration (i)")
plt.ylabel("value of $z(c(x_{i}, y_{i}))$")
plt.title("$T_0$ = " + str(T) + ", f = " + str(f_t))
plt.show()
print(x_coord[-1])
print(y_coord[-1])
print(z[-1])

x = np.arange(-5.0, 5.0, 0.1)
y = np.arange(-5.0, 5.0, 0.1)
X, Y = meshgrid(x, y)  #  grid met punten - origineel
Z = fnc.z_func(X, Y)  #Evaluatie van z op de grid
Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')
plt.plot(x_coord, y_coord, '-.r', markersize = 10, linewidth=1)
plt.plot(x_coord[-1], y_coord[-1], 'oy', markersize = 5)
plt.xlim(-5.0, 5.0)
plt.ylim(-5.0, 5.0)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""