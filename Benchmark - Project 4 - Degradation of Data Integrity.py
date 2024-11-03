# Programmers: Grant Burk and Ben Croyle
#
# Code packages: numpy, scipy, and matplotlib (see below)
#
# The approach to implement was very simple, just define the expression that we got from doing the math by hand and then get a line space then plot the graph.

import numpy as np
import matplotlib.pyplot as plt


# Define the function
def x(t):
    return np.exp(-t/20), -np.exp(-t/20)

# Create time values
t = np.linspace(0, 100, 400)

# Get the corresponding x(t) values
x1, x2 = x(t)

# Plotting the functions
plt.plot(t, x1, label='$e^{-t/20}$')
plt.plot(t, x2, label='$-e^{-t/20}$')

# Add labels and title
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Plot of $x(t)=[e^{-t/20}, -e^{-t/20}]$')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define the matrix A
A = np.array([[-1/50, 3/100], [1/50, -3/100]])

# Define the initial conditions
x0 = np.array([1, -1])

# Define time points
t = np.linspace(0, 100, 400)

# Calculate e^At for each time point
x = np.array([expm(A * ti).dot(x0) for ti in t])

# Extract the solutions x1(t) and x2(t)
x1 = x[:, 0]
x2 = x[:, 1]

# Plotting the solutions
plt.figure()
plt.plot(t, x1, label='$x_1(t)$')
plt.plot(t, x2, label='$x_2(t)$')
plt.xlabel('t')
plt.ylabel('$x(t)$')
plt.title('Solution of the System of Differential Equations')
plt.legend()
plt.grid(True)
