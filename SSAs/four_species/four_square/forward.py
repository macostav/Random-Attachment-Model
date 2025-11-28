import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.integrate import solve_ivp

"""
Experiment where 4-square system has only forward reactions, all with the same rate 1. We notice that
if all monomers start with the same count, then the system is symmetric. We simply to only consider
the following ODEs:

1. dM/dt = -2M^2 -  2MD - MT
2. dD/dt = M^2 - 2MD - D^2
3. dT/dt = 2MD - MT
4. dF/dt = 2D^2 + 4MT

In this script, we solve the above ODEs numerically to inform our predictions about the system.
The labels are:

M: Monomer
D: Dimer
T: Trimer
F: Tetramer
"""

def odes(t,y):
    """
    Return odes for the system.

    :param t: time
    :param y: state vector
    :return: odes
    """
    # State vector
    M, D, T, F = y

    # Individual ODEs
    dM = -2*M**2 -2*M*D - M*T
    dD = M**2 - 2*M*D - 2*D**2
    dT = 2*M*D - M*T
    dF = 1*D**2 + 4*M*T
    
    return [dM, dD, dT, dF]

if __name__ == "__main__":
    # Initial conditions
    species = ["M", "D", "T", "F"]
    y0 = [400, 0, 0, 0]
    duration = 200

    t_span = (0, duration) 
    t_eval = np.linspace(*t_span, 1000) 

    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA')

    # Plotting results
    for i,s in enumerate(species):
        plt.figure(figsize=(6,3))
        plt.plot(sol.t, sol.y[i], '--')
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"forward_{s}.png", dpi=200)
        plt.close()

    # Find proportions of each species
    Mfinal, Dfinal, Tfinal, Ffinal = sol.y[:, -1]
    total_mass = sol.y[:, 0].sum()

    print(f"""Final Proportions:
    Monomer:  {Mfinal/total_mass:.4f}
    Dimer:    {Dfinal/total_mass:.4f}
    Trimer:   {Tfinal/total_mass:.4f}
    Tetramer: {Ffinal/total_mass:.4f}
    """)
    
    # Suppose Mfinal, Dfinal, Tfinal, Ffinal are the *counts of molecules* of each size.
    # If instead they are already mass (monomer-units), set counts_are_molecules = False.
    counts_are_molecules = True


    if counts_are_molecules:
        mass_monomer = Mfinal * 1
        mass_dimer   = Dfinal * 2
        mass_trimer  = Tfinal * 3
        mass_tetramer= Ffinal * 4
    else:
        # they are already mass
        mass_monomer, mass_dimer, mass_trimer, mass_tetramer = Mfinal, Dfinal, Tfinal, Ffinal

    total_mass = mass_monomer + mass_dimer + mass_trimer + mass_tetramer

    pM = mass_monomer / total_mass
    pD = mass_dimer   / total_mass
    pT = mass_trimer  / total_mass
    pF = mass_tetramer/ total_mass

    # print nicely (use triple-quoted f-string to allow line breaks)
    print(f"""Final Proportions (mass-weighted):
    Monomer:  {pM:.6f}
    Dimer:    {pD:.6f}
    Trimer:   {pT:.6f}
    Tetramer: {pF:.6f}

    Sum of proportions = {pM + pD + pT + pF:.12f}
    Total mass = {total_mass:.6f}
    """)