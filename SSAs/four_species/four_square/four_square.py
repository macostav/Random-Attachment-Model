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

We add the case where clusters can start to form. So we now have in total
13 species: {A,B,C,D,AB,AC,BD,CD,ABD,BDC,DCA,CAB,ABCD}.

We work in the case where only pairs can form. We consider 8 species:
{A, B, C, D, AB, AC, BD, CD}. This gives us 4 reversible reactions,
and therefore 8 rates to consider. We list reactions:

monomer + monomer <-> dimer:
1. A + B -> AB
2. AB -> A + B
3. A + C -> AC
4. AC -> A + C
5. B + D -> BD
6. BD -> B + D
7. C + D -> CD
8. CD -> C + D

monomer + dimer <-> trimer:
9. AB + D -> ABD
10. ABD -> AB + D
11. A + BD -> ABD
12. ABD -> A + BD
13. BD + C -> BDC
14. BDC -> BD + C
15. B + CD -> BDC
16. BDC -> B + CD
17. DC + A -> DCA
18. DCA -> DC + A
19. AC + D -> DCA
20. DCA -> AC + D
21. CA + B -> CAB
22. CAB -> CA + B
23. AB + C -> CAB
24. CAB -> AB + C

dimer + dimer <-> tetramer:
25. AB + CD -> ABCD
26. ABCD -> AB + CD
27. AC + BD -> ABCD
28. ABCD -> AC + BD

trimer + monomer <-> tetramer:
29. ABD + C -> ABCD
30. ABCD -> ABD + C
31. BDC + A -> ABCD
32. ABCD -> BDC + A
33. DCA + B -> ABCD
34. ABCD -> DCA + B
35. CAB + D -> ABCD
36. ABCD -> CAB + D
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

species = [
    "A", "B", "C", "D",        # monomers
    "AB", "AC", "BD", "CD",    # dimers
    "ABD", "BDC", "DCA", "CAB",# trimers
    "ABCD"                     # tetramer
]
n_species = len(species)
idx = {s: i for i, s in enumerate(species)}

# Rate dictionary (k1..k36)
rates = {
    "k1": 0.05,  "k2": 0.25,   # A + B <-> AB
    "k3": 0.04,  "k4": 0.22,   # A + C <-> AC
    "k5": 0.05,  "k6": 0.07,   # B + D <-> BD
    "k7": 0.12,  "k8": 0.32,   # C + D <-> CD

    "k9":  0.05, "k10": 0.25,  # AB + D <-> ABD
    "k11": 0.04, "k12": 0.22,  # A + BD <-> ABD
    "k13": 0.05, "k14": 0.07,  # BD + C <-> BDC
    "k15": 0.12, "k16": 0.32,  # B + CD <-> BDC
    "k17": 0.05, "k18": 0.25,  # CD + A <-> DCA
    "k19": 0.04, "k20": 0.22,  # AC + D <-> DCA
    "k21": 0.05, "k22": 0.07,  # AC + B <-> CAB
    "k23": 0.12, "k24": 0.32,  # AB + C <-> CAB

    "k25": 0.95, "k26": 0.05,  # AB + CD <-> ABCD
    "k27": 0.94, "k28": 0.02,  # AC + BD <-> ABCD

    "k29": 0.85, "k30": 0.05,  # ABD + C <-> ABCD
    "k31": 0.84, "k32": 0.02,  # BDC + A <-> ABCD
    "k33": 0.85, "k34": 0.07,  # DCA + B <-> ABCD
    "k35": 0.82, "k36": 0.12   # CAB + D <-> ABCD
}

# ---------------------------
# Reaction definitions (list of dicts)
# reactants and products are dicts: species->stoich (usually 1)
# Rate keys refer to rates dict above.
# ---------------------------
reactions = [
    # monomer + monomer <-> dimer
    {"reactants": {"A":1, "B":1}, "products": {"AB":1}, "k": "k1"},
    {"reactants": {"AB":1},       "products": {"A":1, "B":1}, "k": "k2"},

    {"reactants": {"A":1, "C":1}, "products": {"AC":1}, "k": "k3"},
    {"reactants": {"AC":1},       "products": {"A":1, "C":1}, "k": "k4"},

    {"reactants": {"B":1, "D":1}, "products": {"BD":1}, "k": "k5"},
    {"reactants": {"BD":1},       "products": {"B":1, "D":1}, "k": "k6"},

    {"reactants": {"C":1, "D":1}, "products": {"CD":1}, "k": "k7"},
    {"reactants": {"CD":1},       "products": {"C":1, "D":1}, "k": "k8"},

    # monomer + dimer <-> trimer (8 pairs -> 16 reactions)
    {"reactants": {"AB":1, "D":1}, "products": {"ABD":1}, "k": "k9"},
    {"reactants": {"ABD":1},       "products": {"AB":1, "D":1}, "k": "k10"},

    {"reactants": {"A":1, "BD":1}, "products": {"ABD":1}, "k": "k11"},
    {"reactants": {"ABD":1},       "products": {"A":1, "BD":1}, "k": "k12"},

    {"reactants": {"BD":1, "C":1}, "products": {"BDC":1}, "k": "k13"},
    {"reactants": {"BDC":1},       "products": {"BD":1, "C":1}, "k": "k14"},

    {"reactants": {"B":1, "CD":1}, "products": {"BDC":1}, "k": "k15"},
    {"reactants": {"BDC":1},       "products": {"B":1, "CD":1}, "k": "k16"},

    {"reactants": {"CD":1, "A":1}, "products": {"DCA":1}, "k": "k17"},
    {"reactants": {"DCA":1},       "products": {"CD":1, "A":1}, "k": "k18"},

    {"reactants": {"AC":1, "D":1}, "products": {"DCA":1}, "k": "k19"},
    {"reactants": {"DCA":1},       "products": {"AC":1, "D":1}, "k": "k20"},

    {"reactants": {"AC":1, "B":1}, "products": {"CAB":1}, "k": "k21"},
    {"reactants": {"CAB":1},       "products": {"AC":1, "B":1}, "k": "k22"},

    {"reactants": {"AB":1, "C":1}, "products": {"CAB":1}, "k": "k23"},
    {"reactants": {"CAB":1},       "products": {"AB":1, "C":1}, "k": "k24"},

    # dimer + dimer <-> tetramer
    {"reactants": {"AB":1, "CD":1}, "products": {"ABCD":1}, "k": "k25"},
    {"reactants": {"ABCD":1},       "products": {"AB":1, "CD":1}, "k": "k26"},

    {"reactants": {"AC":1, "BD":1}, "products": {"ABCD":1}, "k": "k27"},
    {"reactants": {"ABCD":1},       "products": {"AC":1, "BD":1}, "k": "k28"},

    # trimer + monomer <-> tetramer (8 reactions)
    {"reactants": {"ABD":1, "C":1}, "products": {"ABCD":1}, "k": "k29"},
    {"reactants": {"ABCD":1},       "products": {"ABD":1, "C":1}, "k": "k30"},

    {"reactants": {"BDC":1, "A":1}, "products": {"ABCD":1}, "k": "k31"},
    {"reactants": {"ABCD":1},       "products": {"BDC":1, "A":1}, "k": "k32"},

    {"reactants": {"DCA":1, "B":1}, "products": {"ABCD":1}, "k": "k33"},
    {"reactants": {"ABCD":1},       "products": {"DCA":1, "B":1}, "k": "k34"},

    {"reactants": {"CAB":1, "D":1}, "products": {"ABCD":1}, "k": "k35"},
    {"reactants": {"ABCD":1},       "products": {"CAB":1, "D":1}, "k": "k36"},
]

n_reactions = len(reactions)
assert n_reactions == 36

# Matrix outlining the stoichiometric changes for each reaction
stoich_changes = np.zeros((n_reactions, n_species), dtype=int)
reactant_lists = []  # list of (species_index, count) tuples for each reaction

for ri, rxn in enumerate(reactions):
    # reactants list
    rlist = []
    for s, cnt in rxn["reactants"].items(): # s = species, cnt = count in a specific reaction
        rlist.append((idx[s], cnt))
    reactant_lists.append(rlist)

    # Writing down how each reaction changes species counts
    for s, cnt in rxn.get("products", {}).items():
        stoich_changes[ri, idx[s]] += cnt
    for s, cnt in rxn.get("reactants", {}).items():
        stoich_changes[ri, idx[s]] -= cnt

# Calculating propensities
def compute_propensities(counts, reactant_lists, rates, reactions):
    """
    counts: 1D array of species counts
    reactant_lists: list of lists of (species_index, stoich)
    rates: dict of k's
    reactions: original reactions list (to get k key)
    """
    a = np.zeros(len(reactant_lists), dtype=float)
    for i, rlist in enumerate(reactant_lists):
        k_key = reactions[i]["k"]
        k = rates[k_key] # Get rate constant

        # If no reactants (shouldn't happen here), propensity = k
        if len(rlist) == 0:
            a[i] = k
            continue

        term = 1.0
        feasible = True

        for s_idx, _ in rlist:  # reactant stoich is always 1
            n = counts[s_idx]
            if n < 1:
                feasible = False
                break
            term *= n # Account for multiple reactants

        a[i] = k * term if feasible else 0.0

    return a

# Gillespie Algorithm
def gillespie_ssa(initial_counts, t_max, reactions, reactant_lists, stoich_changes, rates,
                  max_steps=int(1e7)):
    """
    Runs SSA until t_max or max_steps.
    Returns times array and history dict mapping species->list
    """
    counts = np.array(initial_counts, dtype=int)
    t = 0.0

    # history
    history = {s: [counts[idx_s]] for idx_s, s in enumerate(species)}
    times = [t]

    for step in range(max_steps):
        a = compute_propensities(counts, reactant_lists, rates, reactions)
        a0 = a.sum()
        if a0 <= 0.0:
            # no more reactions possible
            break

        # draw two random numbers
        r1 = random.random()
        r2 = random.random()

        # time to next reaction
        tau = -math.log(r1) / a0
        t += tau

        if t > t_max:
            # we stop; do not apply reaction that would go beyond t_max
            break

        # choose reaction by cumulative sum
        cum = np.cumsum(a) # [a1, a1+a2, a1+a2+a3, ...]
        target = r2 * a0 # target ~ Unif(0,a0)
        ri = np.searchsorted(cum, target) # Reaction index

        # update species counts
        counts += stoich_changes[ri]

        # record
        for idx_s, s in enumerate(species):
            history[s].append(int(counts[idx_s]))
        times.append(t)

    return np.array(times), history

if __name__ == "__main__":
    # initial counts: set these as you like
    initial_counts = np.zeros(n_species, dtype=int)
    initial_counts[idx["A"]] = 50
    initial_counts[idx["B"]] = 50
    initial_counts[idx["C"]] = 50
    initial_counts[idx["D"]] = 50
    # all complexes start at 0

    duration = 100.0  # simulation time

    times, history = gillespie_ssa(initial_counts, duration, reactions,
                                   reactant_lists, stoich_changes, rates)

    for s in species:
        plt.figure(figsize=(6,3))
        plt.step(times, history[s], where="post", label=f"{s} (SSA)")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s} (SSA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"full_square_{s}.png", dpi=200)
        plt.close()

    print("SSA finished.")



