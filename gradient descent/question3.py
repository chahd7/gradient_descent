import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define functions g(x) and h(y)
def g(x):
    return (np.sin(x) - np.sin(2*x)/2 + np.sin(3*x)/3 - np.sin(4*x)/4) * (x**2 / (x + 1))

def h(y):
    return 2 + np.cos(y) + np.cos(2*y - 1/2) / 2

# Create a grid of x and y values
x = np.linspace(-20, 20, 400)
y = np.linspace(-20, 20, 400)
X, Y = np.meshgrid(x, y)

# Compute g(x) and h(y) for each point on the grid
G = g(X)
H = h(Y)

# Compute f(x, y) as the product of g(x) and h(y)
F = G * H



# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, F, cmap='viridis')
fig.colorbar(surf)
ax.set_title('Representation of f(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()


