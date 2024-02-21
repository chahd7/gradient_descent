# Define the function and its gradient
def function(x):
    return (3*x + 5)**2

def gradient(x):
    return 2 * (3*x + 5) * 3

# Initialize variables
cur_x = 3  # Starting point
rate = 0.01  # Learning rate
max_iters = 10000  # Maximum number of iterations
iters = 0  # Iteration counter
precision = 0.0000001

# Gradient descent algorithm
while iters < max_iters:
    prev_x = cur_x  # Store the current value of x
    cur_x = cur_x - rate * gradient(prev_x)  # Update x using the gradient formula
    iters += 1  # Increment the iteration counter

    # Print current point (x, y) and gradient
    print(f"Iteration {iters}: Point ({cur_x:.8f}, {function(cur_x):.8f}), Gradient: {gradient(cur_x)}")

