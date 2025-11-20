import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
Testing SSA from paper. The structure we study is the following:

A - B

We work in the case where only pairs can form. We consider 3 species:
{A, B, AB}. This gives us 1 reversible reaction, and thus 2 rates to consider.
"""

def ODE_solution_AB(nA, nB, nAB, k_1, k_2, t):
    """
    Writes equation of the ODE solution for the system. The solution comes from
    solving the deterministic rate equation for AB analytically.

    :param nA, nB, nAB: initial number of each species
    :param k_1, k_2: reaction rates
    :param t: time
    """
    # Constants that help express the solution more compactly
    S = nA + nB + 2*nAB + k_2/k_1
    r_1 = (S - np.sqrt(S**2 - 4*(nA+nAB)*(nB + nAB)))/2    
    r_2 = (S + np.sqrt(S**2 - 4*(nA+nAB)*(nB + nAB)))/2 
    
    K = (nAB - r_1)/(nAB-r_2)
    U = k_1*(r_1-r_2)

    return (r_1 - r_2*K*np.exp(U*t))/(1-K*np.exp(U*t))

def deterministic_arrays(nA, nB, nAB, k_1, k_2, array_t):
    """
    Creating arrays for the solutions to deterministic ODEs.

    :param nA, nB, nAB: initial number of each species
    :param k_1, k_2: reaction rates
    :param array_t: array of time points
    """
    deterministic_A = []
    deterministic_B = []
    deterministic_AB = []

    for t in array_t:
        # Computing the deterministic solution at time t
        nAB_t = ODE_solution_AB(nA, nB, nAB, k_1, k_2, t)
        nA_t = (nA+nAB) - nAB_t
        nB_t = (nB+nAB) - nAB_t

        # Adding elements to the array
        deterministic_AB.append(nAB_t)
        deterministic_A.append(nA_t)
        deterministic_B.append(nB_t)
    return deterministic_A, deterministic_B, deterministic_AB


def next_reaction():
    """
    Generate two random numbers. One tells you the time until the next reaction,
    the other one tells you which reaction happens.

    :return: time for next reaction, reaction type
    """
    r_1 = random.random()
    r_2 = random.random()

    return r_1, r_2

def propensity(k_1, k_2, nA, nB, nAB):
    """
    Compute the propensities for the two possible reactions.

    :param k_1, k_2: reaction rates
    :param nA, nB, nAB: current number of each species
    :return: propensity for 1, propensity for 2
    """
    a1 = k_1 * nA * nB
    a2 = k_2 * nAB
    return a1, a2

def plot_all_results(k_1, k_2, array_A, array_B, array_AB, deterministic_A, deterministic_B, deterministic_AB, array_t):
    """
    Plot all the results of the simulation.

    :param array_A, array_B, array_AB: arrays with number of species
    :param deterministic_A, deterministic_B, deterministic_AB: arrays with determinisic solutions
    :param array_t: array keeping track of time
    """
    plt.figure()  # Start a new figure

    plt.plot(array_t, array_A, label="A")
    plt.plot(array_t, array_B, label="B")
    plt.plot(array_t, array_AB, label="AB")
    plt.plot(array_t, deterministic_A, label="det_A")
    plt.plot(array_t, deterministic_B, label="det_B")
    plt.plot(array_t, deterministic_AB, label="det_AB")

    # Labels
    plt.title(f"kon = {k_1}, koff = {k_2}")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Species")
    plt.legend()
    plt.savefig("two_species_CHECK.png")
    #plt.show()

def plot_species(species, k_1, k_2, array_species, deterministic, array_t):
    """
    Plots the results of only one species.

    :param species: which species to plot ("A", "B", or "AB")
    :param k_1, k_2: reaction rates
    :param array_species: array with the number of species from simulation
    :param deterministic: solution to determinisic ODE for species
    :param array_t: array of time points
    """
    plt.figure()  # Start a new figure

    plt.plot(array_t, array_species, label="SSA")
    plt.plot(array_t, deterministic, label="ODE", linewidth =1)

    # Labels
    plt.title(f"Species {species} with kon = {k_1}, koff = {k_2}")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Species")
    plt.legend()
    plt.savefig(f"two_species_{species}.png")

if __name__ == "__main__":
    # Starting parameters
    nA = 5000 # Initial number of A
    nB = 3000 # Initial number of B
    nAB = 2000 # Initial number of AB

    k_1 = 0.05 # Rate A + B -> AB
    k_2 = 0.15 # Rate AB -> A + B

    t = 0 # Initial time
    duration = 1 # Duration of simulation

    # Initialize arrays
    array_A = [nA]
    array_B = [nB]
    array_AB = [nAB]
    array_t = [t]

    while t < duration:
        # Determine which reaction is to happen
        r_1, r_2 = next_reaction()
        a1, a2 = propensity(k_1, k_2, nA, nB, nAB)
        a0 = a1 + a2 # Overall propensity

        tau = 1/a0 * math.log(1/r_1) # Time for next reaction

        t  += tau

        if r_2 <= a1/a0:
            # A + B -> AB
            nA -= 1
            nB -= 1
            nAB += 1
        else:
            # AB -> A + B
            nA += 1
            nB += 1
            nAB -= 1

        array_A.append(nA)
        array_B.append(nB)
        array_AB.append(nAB)
        array_t.append(t)
    
    # Plot results
    deterministic_A, deterministic_B, deterministic_AB = deterministic_arrays(array_A[0], array_B[0], array_AB[0], k_1, k_2, array_t)
    plot_all_results(k_1, k_2, array_A, array_B, array_AB, deterministic_A, deterministic_B, deterministic_AB, array_t)
    plot_species("A", k_1, k_2, array_A, deterministic_A, array_t)
    plot_species("B", k_1, k_2, array_B, deterministic_B, array_t)
    plot_species("AB", k_1, k_2, array_AB, deterministic_AB, array_t)





