# Code by Grant Burk, modified by Ben Croyle, dependencies / packages used include NumPy, SciPy, and matplotlib, and these were used to solve the ODE for Newton's Law of Cooling by first defining the ODE with parameters and variables before using a function to solve them.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE
def cooling_rate(T, t, k, T_ambient):
    return -k*(T - T_ambient) # Newton's Law of Cooling formula
    # TODO:  explain the derivative in the documentation and what it represents
    # TODO:  No need to solve the equation in the doc, just explain its purpose and parts

# Parameters
H = 150 # W / m^2, Heat Transfer Rate for Aluminum
A = 0.001 # m^2, the Estimated Surface Area of a CPU
k = 0.1  # Cooling rate constant
T_ambient = 20  # Ambient temperature
T_initial = 100  # Initial temperature

# Time points
t = np.linspace(0, 60, 100)  # Time in minutes

# Solve the ODE
T = odeint(cooling_rate, T_initial, t, args=(H * A, T_ambient))

# Plot the solution
plt.plot(t, T)
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature of a Cooling Object")
plt.grid(True)
plt.show()