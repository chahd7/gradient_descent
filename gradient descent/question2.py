# Define the function and its gradient
def func(x):
    return (3*x + 5)**2

def grad(x):
    return 2 * (3*x + 5) * 3

# Initialize variables
cur_x = 3
rate = 0.01  # Learning rate
precision = 0.0000001  # This tells us when to stop the algorithm
max_iters = 10000  # Maximum number of iterations
iters = 0  # Iteration counter

# Gradient descent algorithm
while iters < max_iters:
    prev_x = cur_x #store the current value of x
    cur_x = cur_x - rate * grad(prev_x) #update x using the gradient formula 
    iters += 1 #increment the iteration counter
    # Print current point (x, y) and gradient
    print(f"Iteration {iters}: Point ({cur_x:.8f}, {func(cur_x):.8f}), Gradient: {grad(cur_x)}")
    # Check for convergence
    if abs(cur_x - prev_x) < precision: #check if change in x is smaller than precision
        print("Gradient descent converged.")
        break