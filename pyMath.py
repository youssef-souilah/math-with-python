import matplotlib.pyplot as plt
import numpy as np
import math


x = np.linspace(1, 20, 50)
y = np.linspace(1, 20, 50)

y, x = np.meshgrid(y, x) 

def delCAD(x, y):
    return np.sqrt((2*x)/np.exp(-y))

def plot(a, b):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 6))
    surf = ax.plot_trisurf(x.flat, y.flat, delCAD(x, y).flat, cmap='coolwarm')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=160)
    plt.show()
    return

plot(12,10)