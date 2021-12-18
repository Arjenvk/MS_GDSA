import math
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sympy as sym
import random


## functies behorende bij main bestand

# hoofd functie
def z_func(x, y):
    return -np.cos(x) * np.cos(y) * (np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) + 1.5*(np.exp(-0.1*((x+np.pi)**2 + (y+np.pi)**2))))
# partitieel afgeleiden functies
def z_func_diffx(x,y):
    return np.cos(y) * (np.sin(x) * (np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) + 1.5*np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2))) + 0.2 * np.cos(x) * (np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) * (x-np.pi) + 1.5*np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) * (x+np.pi)))
def z_func_diffy(x,y):
    return np.cos(x) * (np.sin(y) * (np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) + 1.5*np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2))) + 0.2 * np.cos(y) * (np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) * (y-np.pi) + 1.5*np.exp(-0.1*((x-np.pi)**2 + (y-np.pi)**2)) * (y+np.pi)))

# functie om 2d plot te maken
def plot2d(X,Y,Z):
    Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()
    c = plt.imshow(Z, cmap='viridis', vmin=Z_min, vmax=Z_max,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation='nearest', origin='lower')
    plt.colorbar(c)
    plt.title('2d presentation of the function',
              fontweight="bold")
    plt.show()


# functie voor 3d plot
def plot3d(X,Y,Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# functie om volgende punt in GD te bepalen
def GD(x,y, num_iter, alpha):
    x_coord = [x]
    y_coord = [y]
    x_diff = []
    y_diff = []
    for iter in range(num_iter):
        # bepaal partitieel afgeleide in punt
        dx = z_func_diffx(x_coord[iter], y_coord[iter])
        x_diff.append(dx)
        dy = z_func_diffy(x_coord[iter], y_coord[iter])
        y_diff.append(dx)
        # maak volgend punt adhv partitieel afgeleiden en tuning parameter alpha
        x_nieuw = x_coord[iter] + alpha * dx
        y_nieuw = y_coord[iter] + alpha * dy
        # voef punt toe aan lijst
        x_coord.append(x_nieuw)
        y_coord.append(y_nieuw)
    return(x_coord, y_coord, x_diff, y_diff)

# functie om volgende punt in GD te bepalen met dynamische alpha
def GD_alpha(x,y, num_iter, alpha, factor):
    x_coord = [x]
    y_coord = [y]
    x_diff = []
    y_diff = []
    for iter in range(num_iter):
        # bepaal partitieel afgeleide in punt
        dx = z_func_diffx(x_coord[iter], y_coord[iter])
        x_diff.append(dx)
        dy = z_func_diffy(x_coord[iter], y_coord[iter])
        y_diff.append(dx)
        # alpha bepalen
        alpha = alpha * factor**iter
        # maak volgend punt adhv partitieel afgeleiden en tuning parameter alpha
        x_nieuw = x_coord[iter] + alpha * dx
        y_nieuw = y_coord[iter] + alpha * dy
        # voef punt toe aan lijst
        x_coord.append(x_nieuw)
        y_coord.append(y_nieuw)
    return(x_coord, y_coord, x_diff, y_diff)

# functie om nieuwe buur te maken
def func_buur(x,y, sigma):
    x2 = x + np.random.normal(0, sigma)
    y2 = y + np.random.normal(0, sigma)
    return x2, y2

def func_SA(x_oud, y_oud, sigma, T, factor, num_iter):
    z_oud = z_func(x_oud, y_oud)                        # evalueer oude coordinaat
    x_nieuw, y_nieuw = func_buur(x_oud, y_oud, sigma)   # maak nieuwe buur
    x_nieuw = func_checkdomain(x_nieuw)                 # check of x-coord in domein ligt, anders spiegel naar binnen
    y_nieuw = func_checkdomain(y_nieuw)                 # check of y coord in domein ligt, anders spiegel naar binnen
    z_nieuw = z_func(x_nieuw, y_nieuw)                  # evalueer nieuwe coordinaat
    if z_nieuw <= z_oud:
        z_uit = z_nieuw
        x_uit = x_nieuw
        y_uit = y_nieuw
    else:
        T = T * factor**num_iter
        p = np.exp((z_oud-z_nieuw)/T)
        k = np.random.uniform(0,1)
        if k <= p:
            z_uit = z_nieuw
            x_uit = x_nieuw
            y_uit = y_nieuw
        else:
            z_uit = z_oud
            x_uit = x_oud
            y_uit = y_oud
    return(x_uit, y_uit, z_uit)

def func_checkdomain(coord):
    if abs(coord) > 5:
        if coord < 0:
            coord2 = coord + 2* (abs(coord)-5)
        else:
            coord2 = coord - 2*(abs(coord)-5)
    else:
        coord2 = coord
    return(coord2)

def func_SA_seed(x_oud, y_oud, sigma, T, factor, num_iter):
    z_oud = z_func(x_oud, y_oud)                        # evalueer oude coordinaat
    x_nieuw, y_nieuw = func_buur_seed(x_oud, y_oud, sigma,num_iter)   # maak nieuwe buur
    x_nieuw = func_checkdomain(x_nieuw)                 # check of x-coord in domein ligt, anders spiegel naar binnen
    y_nieuw = func_checkdomain(y_nieuw)                 # check of y coord in domein ligt, anders spiegel naar binnen
    z_nieuw = z_func(x_nieuw, y_nieuw)                  # evalueer nieuwe coordinaat
    if z_nieuw <= z_oud:
        z_uit = z_nieuw
        x_uit = x_nieuw
        y_uit = y_nieuw
    else:
        T = T * (factor**num_iter)
        p = np.exp((z_oud-z_nieuw)/T)
        np.random.seed(num_iter)
        k = np.random.uniform(0,1)
        if k <= p:
            z_uit = z_nieuw
            x_uit = x_nieuw
            y_uit = y_nieuw
        else:
            z_uit = z_oud
            x_uit = x_oud
            y_uit = y_oud
    return(x_uit, y_uit, z_uit)

def func_buur_seed(x,y, sigma, num_iter):
    np.random.seed(num_iter)
    x2 = x + np.random.normal(0, sigma)
    y2 = y + np.random.normal(0, sigma)
    return x2, y2


def func_SA_GD(x_oud, y_oud, sigma, T, f_t, num_iter, f_gd, alpha):
    z_oud = z_func(x_oud, y_oud)  # evalueer oude coordinaat
    x_nieuw, y_nieuw = func_buur_GD(x_oud, y_oud, sigma, f_gd, num_iter, alpha)  # maak nieuwe buur
    x_nieuw = func_checkdomain(x_nieuw)  # check of x-coord in domein ligt, anders spiegel naar binnen
    y_nieuw = func_checkdomain(y_nieuw)  # check of y coord in domein ligt, anders spiegel naar binnen
    z_nieuw = z_func(x_nieuw, y_nieuw)  # evalueer nieuwe coordinaat
    if z_nieuw <= z_oud:
        z_uit = z_nieuw
        x_uit = x_nieuw
        y_uit = y_nieuw
    else:
        T = T * f_t ** num_iter
        p = np.exp((z_oud - z_nieuw) / T)
        k = np.random.uniform(0, 1)
        if k <= p:
            z_uit = z_nieuw
            x_uit = x_nieuw
            y_uit = y_nieuw
        else:
            z_uit = z_oud
            x_uit = x_oud
            y_uit = y_oud
    return (x_uit, y_uit, z_uit)


def func_buur_GD(x,y, sigma, f_gd, num_iter, alpha):
    p = f_gd**num_iter
    k = np.random.uniform(0, 1)
    if k <= p:
        x2 = x + np.random.normal(0, sigma)
        y2 = y + np.random.normal(0, sigma)
    else:
        dx = z_func_diffx(x, y)
        dy = z_func_diffy(x, y)
        x2 = x + alpha * dx
        y2 = y + alpha * dy
    return x2, y2