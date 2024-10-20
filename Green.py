# Code by Grant Burke and Ben Croyle. The following packages used were numpy, scipy, and matplotlib. THe approach taken was to first solve the homogeneous forms of the two equations, and then solve them using Green's function. To do so, we used scipy.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import quad

# Define the ODE system (y''+y=4, homogeneous)
def odefunc(t, y):
    dydt = [y[1], -y[0] + 4]
    return dydt

# Initial conditions
y0 = [0, 0]  # y(0) = 0, y'(0) = 0

# Time points
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the ODE
solution = solve_ivp(odefunc, t_span, y0, t_eval=t_eval)

# Extract solution
t = solution.t
y = solution.y[0]

# Plot the solution
plt.plot(t, y, label='y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' + y = 4 with initial conditions y(0) = 0, y\'(0) = 0')
plt.grid(True)
plt.show()

# Define the ODE system (y''+4y=x^2, homogeneous)
def odefunc(x, y):
    dydx = [y[1], x**2 - 4*y[0]]
    return dydx

# Initial conditions
y0 = [0, 0]  # y(0) = 0, y'(0) = 0

# Time points
x_span = (0, 10)
x_eval = np.linspace(x_span[0], x_span[1], 500)

# Solve the ODE
solution = solve_ivp(odefunc, x_span, y0, t_eval=x_eval)

# Extract solution
x = solution.t
y = solution.y[0]

# Plot the solution
plt.plot(x, y, label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' + 4y = x^2 with initial conditions y(0) = 0, y\'(0) = 0')
plt.grid(True)
plt.show()

# Define the Green's function (y''+y=4)
def G(t, tau):
    return np.where(tau <= t, np.sin(t - tau), 0)

# Define the integrand for the particular solution
def integrand(tau, t):
    return G(t, tau) * 4

# Define the function to calculate the particular solution y_p(t)
def y_p(t):
    result, _ = quad(integrand, 0, t, args=(t,))
    return result

# Define the general solution y(t)
def y(t):
    return y_p(t)

# Set up the points to evaluate the solution
t_values = np.linspace(0, 10, 500)
y_values = np.array([y(t) for t in t_values])

# Plot the solution
plt.plot(t_values, y_values, label='$y(t)$')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.title('Solution of $y\'\' + y = 4$ with initial conditions $y(0) = 0$, $y\'(0) = 0$')
plt.grid(True)
plt.show()

# Define the Green's function (y''+4y=x^2)
def G(tau, t):
    return np.where(tau <= t, (np.sin(2*(t-tau)) / 2), 0)

# Define the integrand for the particular solution
def integrand(tau, t):
    return G(tau, t) * tau**2

# Define the function to calculate the particular solution y_p(t)
def y_p(t):
    result, _ = quad(integrand, 0, t, args=(t,))
    return result

# Define the general solution y(t)
def y(t):
    return y_p(t)

# Set up the points to evaluate the solution
t_values = np.linspace(0, 10, 500)
y_values = np.array([y(t) for t in t_values])

# Plot the solution
plt.plot(t_values, y_values, label='$y(t)$')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.title('Solution of $y\'\' + 4y = x^2$ with initial conditions $y(0) = 0$, $y\'(0) = 0$')
plt.grid(True)
plt.show()
