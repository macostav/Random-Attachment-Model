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
    m_a, m_b, d, t_, q = y

    dm_a = -2*m_a*m_b - 2*m_a*d - m_a*t_
    dm_b = -2*m_b*m_a - 2*m_b*d - m_b*t_

    dd   =  m_a*m_b - d*(m_a+m_b) - d*d

    dt_  = 2*d*(m_a+m_b) - t_*(m_a+m_b)

    dq   = d*d + t_*(m_a+m_b)

    return [dm_a, dm_b, dd, dt_, dq]

if __name__ == "__main__":
    species = ["m_a","m_b","d","t","q"]
    y0 = [200,200,0,0,0]
    duration = 200

    sol = solve_ivp(odes, (0,duration), y0, t_eval=np.linspace(0,duration,1000), method="LSODA")

    for i,s in enumerate(species):
        plt.figure(figsize=(6,3))
        plt.plot(sol.t, sol.y[i])
        plt.title(s)
        plt.tight_layout()
        plt.savefig(f"correct_{s}.png", dpi=200)
        plt.close()

    m_a, m_b, d, t_, q = sol.y[:, -1]

    total_mass = (
        1*m_a +
        1*m_b +
        2*d   +
        3*t_  +
        4*q
    )

    print(f"""
Final proportions (mass fractions):
  m_a: {m_a/total_mass:.4f}
  m_b: {m_b/total_mass:.4f}
  d:   {(2*d)/total_mass:.4f}
  t:   {(3*t_)/total_mass:.4f}
  q:   {(4*q)/total_mass:.4f}

Check: sum = {
    (m_a + m_b + 2*d + 3*t_ + 4*q)/total_mass
}
and {total_mass}
""")
