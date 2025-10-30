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

def plot_results(array_A, array_B, array_AB, array_t):
    """
    Plot the results of the simulation.

    :param array_A, array_B, array_AB: arrays with number of species
    :param array_t: array keeping track of time
    """
    plt.plot(array_t, array_A, label="A")
    plt.plot(array_t, array_B, label="B")
    plt.plot(array_t, array_AB, label="AB")

    # Labels
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Species")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Starting parameters
    nA = 50 # Initial number of A
    nB = 30 # Initial number of B
    nAB = 20 # Initial number of AB

    k_1 = 0.01 # Rate A + B -> AB
    k_2 = 0.45 # Rate AB -> A + B

    t = 0 # Initial time
    duration = 100 # Duration of simulation

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
    plot_results(array_A, array_B, array_AB, array_t)





