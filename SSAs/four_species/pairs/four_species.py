import math
import random
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Testing SSA from paper. The structure we study is the following:

A — B
|   |
C - D

We work in the case where only pairs can form. We consider 8 species:
{A, B, C, D, AB, AC, BD, CD}. This gives us 4 reversible reactions,
and therefore 8 rates to consider. We list reactions:

1. A + B -> AB
2. AB -> A + B
3. A + C -> AC
4. AC -> A + C
5. B + D -> BD
6. BD -> B + D
7. C + D -> CD
8. CD -> C + D
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

def propensity(k, n1, n2):
    """
    Compute the propensities for the two possible reactions.

    :param k: reaction rate
    :param n1, n2: current number of each species (set n2 = 1 for unimolecular)
    :return: propensity
    """
    prop = k * n1 * n2
    return prop

def odes(t, y, rates):
    """
    Return odes for the four species system.

    :param t: time
    :param y: state vector
    :param rates: reaction rates
    """
    A, B, C, D, AB, AC, BD, CD = y # State vector

    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = rates

    # Reaction rates
    r1f = k1f * A * B
    r1r = k1r * AB

    r2f = k2f * A * C
    r2r = k2r * AC

    r3f = k3f * B * D
    r3r = k3r * BD

    r4f = k4f * C * D
    r4r = k4r * CD

    # ODEs
    dA  = -r1f + r1r - r2f + r2r
    dB  = -r1f + r1r - r3f + r3r
    dC  = -r2f + r2r - r4f + r4r
    dD  = -r3f + r3r - r4f + r4r

    dAB =  r1f - r1r
    dAC =  r2f - r2r
    dBD =  r3f - r3r
    dCD =  r4f - r4r

    return [dA, dB, dC, dD, dAB, dAC, dBD, dCD]

if __name__ == "__main__":
    # Starting parameters
    nA = 50 # Initial number of A
    nB = 50 # Initial number of B
    nC = 10 # Initial number of C
    nD = 25 # Initial number of D
    
    nAB = 0 # Initial number of AB
    nAC = 0 # Initial number of AC
    nBD = 0 # Initial number of BD
    nCD = 0 # Initial number of CD

    k_1 = 0.05 # Rate A + B -> AB
    k_2 = 0.25 # Rate AB -> A + B
    k_3 = 0.04 # Rate A + C -> AC
    k_4 = 0.22 # Rate AC -> A + C
    k_5 = 0.05 # Rate B + D -> BD
    k_6 = 0.07 # Rate BD -> B + D
    k_7 = 0.12 # Rate C + D -> CD
    k_8 = 0.32 # Rate CD -> C + D

    t = 0 # Initial time
    duration = 200 # Duration of simulation

    # Initialize arrays
    array_A = [nA]
    array_B = [nB]
    array_C = [nC]
    array_D = [nD]

    array_AB = [nAB]
    array_AC = [nAC]
    array_BD = [nBD]
    array_CD = [nCD]

    array_t = [t]

    while t < duration:
        # Determine which reaction is to happen
        r_1, r_2 = next_reaction()

        # Compute propensities
        a1 = propensity(k_1, nA, nB)
        a2 = propensity(k_2, nAB, 1)
        a3 = propensity(k_3, nA, nC)
        a4 = propensity(k_4, nAC, 1)
        a5 = propensity(k_5, nB, nD)
        a6 = propensity(k_6, nBD, 1)
        a7 = propensity(k_7, nC, nD)
        a8 = propensity(k_8, nCD, 1)

        a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 # Overall propensity

        tau = 1/a0 * math.log(1/r_1) # Time for next reaction

        t  += tau

        if r_2 <= a1/a0:
            # A + B -> AB
            nA -= 1
            nB -= 1
            nAB += 1
        elif a1/a0 < r_2 <= (a1+a2)/a0:
            # AB -> A + B
            nA += 1
            nB += 1
            nAB -= 1
        elif (a1+a2)/a0 < r_2 <= (a1+a2+a3)/a0:
            # A + C -> AC
            nA -= 1
            nC -= 1
            nAC += 1
        elif (a1+a2+a3)/a0 < r_2 <= (a1+a2+a3+a4)/a0:
            # AC -> A + C
            nA += 1
            nC += 1
            nAC -= 1
        elif (a1+a2+a3+a4)/a0 < r_2 <= (a1+a2+a3+a4+a5)/a0:
            # B + D -> BD
            nB -= 1
            nD -= 1
            nBD += 1
        elif (a1+a2+a3+a4+a5)/a0 < r_2 <= (a1+a2+a3+a4+a5+a6)/a0:
            # BD -> B + D
            nB += 1
            nD += 1
            nBD -= 1
        elif (a1+a2+a3+a4+a5+a6)/a0 < r_2 <= (a1+a2+a3+a4+a5+a6+a7)/a0:
            # C + D -> CD
            nC -= 1
            nD -= 1
            nCD += 1
        else:
            # CD -> C + D
            nC += 1
            nD += 1
            nCD -= 1

        array_A.append(nA)
        array_B.append(nB)
        array_C.append(nC)
        array_D.append(nD)

        array_AB.append(nAB)
        array_AC.append(nAC)
        array_BD.append(nBD)
        array_CD.append(nCD)

        array_t.append(t)

    ### Solving ODEs
    y0 = [array_A[0], array_B[0], array_C[0], array_D[0], array_AB[0], array_AC[0], array_BD[0], array_CD[0]] 
    rates = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8] 
    
    t_span = (0, duration) 
    t_eval = np.linspace(*t_span, 1000) 
    
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA', args=(rates,))
    
    # Plot results
    ssa_data = {
    "A":  array_A,
    "B":  array_B,
    "C":  array_C,
    "D":  array_D,
    "AB": array_AB,
    "AC": array_AC,
    "BD": array_BD,
    "CD": array_CD
    }

    species_labels = ["A","B","C","D","AB","AC","BD","CD"]

    for i, label in enumerate(species_labels):
        plt.figure(figsize=(6,4))

        # plot SSA trajectory
        plt.plot(array_t, ssa_data[label], label=f"{label} (SSA)")

        # plot ODE trajectory
        plt.plot(sol.t, sol.y[i], '--', label=f"{label} (ODE)")

        plt.xlabel("Time (s)")
        plt.ylabel(f"{label} Count")
        plt.title(f"{label}: SSA vs ODE")
        plt.legend()

        # save unique file for each species
        plt.savefig(f"{label}_four_species.png", dpi=200)
        plt.close()


