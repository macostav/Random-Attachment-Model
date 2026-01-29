import numpy as np
import matplotlib.pyplot as plt
from math import comb, factorial
from scipy.integrate import solve_ivp

# CONSTANTS:
N_A = 130
N_B = 100
N = min(N_A, N_B)
energy = -np.log(1/(N_A*N_B)) # INTERESTING: -np.log(1/(30*20))

def microstate_weight(n):
    """Calculate the unnormalized weight of a microstate with n bound pairs."""
    
    return comb(N_A, n) * comb(N_B, n) * factorial(n) * np.exp(-energy*n)

def update_system(n_current):
    """Update the system by randomly choosing to bind or unbind a pair."""
    # We propose a move with equal probability (H_ij = 1/2)
    if n_current == 0:
        proposed_move = n_current + 1 # Can only bind
    elif n_current == N:
        proposed_move = n_current - 1 # Can only unbing
    else:
        step = 1 if np.random.rand() < 0.5 else -1
        proposed_move = n_current + step

    # Acceptance probability
    w_current = microstate_weight(n_current)
    w_proposed = microstate_weight(proposed_move)
    acceptance_prob = min(1, w_proposed / w_current)

    # Accept or reject the move
    if np.random.rand() < acceptance_prob:
        return proposed_move
    else:
        return n_current

# Simulation parameters
num_steps = 1000
n = 0 # Initial number of bound pairs
history = [n]

# Run Metropolis-Hastings
for step in range(num_steps):
    n = update_system(n)
    history.append(n)

history = np.array(history)


def odes(t,y):
    """
    Return odes for the system.

    :param t: time
    :param y: state vector
    :return: odes
    """
    n = y

    dn = (N_A - n) * (N_B - n) - np.exp(energy) * n

    return [dn]

y0 = np.array([0.0])
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA')

plt.plot(sol.t, sol.y[0])
plt.title("ODE solution for number of AB bonds over time")
plt.xlabel("Time")
plt.ylabel("Number of AB bonds (n)")
plt.show()
print(f"ODE steady-state number of bonds: {sol.y[0][-1]}")

# Plotting the results
plt.hist(history, bins=N+1, density=True, alpha=0.6, color='skyblue', edgecolor='black')
plt.xlabel("Number of AB bonds (n)")
plt.ylabel("Probability")
plt.title("Metropolis-Hastings sampling of π(n)")
plt.show()

# Average number of bonds
mean_n = np.mean(history)
print("Average number of AB bonds:", mean_n)
