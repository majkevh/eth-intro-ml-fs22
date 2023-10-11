import numpy as np
import matplotlib.pyplot as plt

# Define the function to be plotted
def objective(alpha, delta):
    return -2 * (alpha**2 - 2 * alpha * delta + delta**2) + delta * (np.log(delta) - np.log(alpha)) + (1 - delta) * (np.log(1 - delta) - np.log(1 - alpha))

# Range of values for alpha and delta
alpha_values = np.linspace(0.001, 0.5, 500)  # Avoiding alpha=0 for log calculations
delta_values = np.linspace(0.001, 0.5, 500)  # Avoiding delta=0 for log calculations

# Initialize arrays to store function values
z_delta = np.zeros_like(delta_values)
z_alpha = np.zeros_like(alpha_values)

# Calculate function values for fixed alpha and delta
fixed_alpha = 0.3  # Choose a fixed alpha value
for i, delta in enumerate(delta_values):
    z_delta[i] = objective(fixed_alpha, delta)

fixed_delta = 0.125  # Choose a fixed delta value
for i, alpha in enumerate(alpha_values):
    z_alpha[i] = objective(alpha, fixed_delta)

# Create 2D plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(delta_values, z_delta)
plt.xlabel('Delta')
plt.ylabel('Objective Function Value')
plt.title(f'Objective Function for Fixed Alpha = {fixed_alpha}')

plt.subplot(1, 2, 2)
plt.plot(alpha_values, z_alpha)
plt.xlabel('Alpha')
plt.ylabel('Objective Function Value')
plt.title(f'Objective Function for Fixed Delta = {fixed_delta}')

plt.tight_layout()
plt.show()
