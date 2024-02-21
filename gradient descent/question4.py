import numpy as np


# Define the functions g(x) and h(y)
def g(x):
    return (np.sin(x) - np.sin(2*x)/2 + np.sin(3*x)/3 - np.sin(4*x)/4) * (x**2 / (x + 1))

def h(y):
    return 2 + np.cos(y) + np.cos(2*y - 1/2) / 2

# Define the domain and sampling rate
sampling_rate = 0.1
x_domain = np.linspace(-20, 20, int(40 / sampling_rate))
y_domain = np.linspace(-20, 20, int(40 / sampling_rate))

# Generate samples
samples = []
for x in x_domain:
    for y in y_domain:
        f = g(x) * h(y)
        samples.append((x, y, f))

# Print samples
for sample in samples:
    print(f"x: {sample[0]:.2f}, y: {sample[1]:.2f}, f(x, y): {sample[2]:.6f}")

print("Total number of samples generated:", len(samples))


def generate_samples():
    return samples