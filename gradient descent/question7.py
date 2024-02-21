import numpy as np
from question6 import generate_samples

# Define the function f(x, y)
def g(x):
    return (np.sin(x) - np.sin(2*x)/2 + np.sin(3*x)/3 - np.sin(4*x)/4) * (x**2 / (x + 1))

def h(y):
    return 2 + np.cos(y) + np.cos(2*y - 1/2) / 2

def func(x, y):
    return g(x) * h(y)

# Define the gradient functions for g(x) and h(y)
def grad_g(x):
    return ((np.cos(x) - 2*np.cos(2*x) + 3*np.cos(3*x) - 4*np.cos(4*x)) * (x**2 / (x + 1)) +
            (2*x**3 / (x + 1)) * (-np.sin(x) - 2*np.sin(2*x) + 3*np.sin(3*x) - 4*np.sin(4*x)))

def grad_h(y):
    return (-(np.sin(y) + np.sin(2*y - 1/2)) +
            (np.sin(2*y - 1/2) / 2))

def gradient_descent(samples):
    # Initialize variables
    rate = 0.01  # Learning rate
    precision = 0.0000001  # This tells us when to stop the algorithm
    max_iters = 10000  # Maximum number of iterations

    # Gradient descent algorithm
    for sample in samples:
        cur_x, cur_y, _ = sample
        iters = 0  # Iteration counter
        while iters < max_iters:
            prev_x, prev_y = cur_x, cur_y  # Store the current values of x and y
            # Update x and y using the gradients
            cur_x = cur_x - rate * grad_g(prev_x)
            cur_y = cur_y - rate * grad_h(prev_y)
            iters += 1  # Increment the iteration counter
            # Check for convergence
            if abs(cur_x - prev_x) < precision and abs(cur_y - prev_y) < precision:
                print(f"Local minimum found at ({cur_x:.8f}, {cur_y:.8f}), f(x, y): {func(cur_x, cur_y):.8f}")
                break
        else:
            print("Gradient descent did not converge for the current sample.")

# Generate samples
samples = generate_samples()

# Apply gradient descent on the generated samples
gradient_descent(samples)