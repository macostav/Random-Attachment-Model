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
13 species: {A,B,C,D,AB,AC,BD,CD,ABD,BCD,ACD,ABC,ABCD}.

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
13. BD + C -> BCD
14. BCD -> BD + C
15. B + CD -> BCD
16. BCD -> B + CD
17. DC + A -> ACD
18. ACD -> DC + A
19. AC + D -> ACD
20. ACD -> AC + D
21. CA + B -> ABC
22. ABC -> CA + B
23. AB + C -> ABC
24. ABC -> AB + C

dimer + dimer <-> tetramer:
25. AB + CD -> ABCD
26. ABCD -> AB + CD
27. AC + BD -> ABCD
28. ABCD -> AC + BD

trimer + monomer <-> tetramer:
29. ABD + C -> ABCD
30. ABCD -> ABD + C
31. BCD + A -> ABCD
32. ABCD -> BCD + A
33. ACD + B -> ABCD
34. ABCD -> ACD + B
35. ABC + D -> ABCD
36. ABCD -> ABC + D
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

species = [
    "A", "B", "C", "D",        # monomers
    "AB", "AC", "BD", "CD",    # dimers
    "ABD", "BCD", "ACD", "ABC",# trimers
    "ABCD"                     # tetramer
]
n_species = len(species)
idx = {s: i for i, s in enumerate(species)}

# Rate dictionary (k1..k36)
rates = {
    "k1": 1,  "k2": 0,   # A + B <-> AB
    "k3": 1,  "k4": 0,   # A + C <-> AC
    "k5": 1,  "k6": 0,   # B + D <-> BD
    "k7": 1,  "k8": 0,   # C + D <-> CD

    "k9":  1, "k10": 0,  # AB + D <-> ABD
    "k11": 1, "k12": 0,  # A + BD <-> ABD
    "k13": 1, "k14": 0,  # BD + C <-> BCD
    "k15": 1, "k16": 0,  # B + CD <-> BCD
    "k17": 1, "k18": 0,  # CD + A <-> ACD
    "k19": 1, "k20": 0,  # AC + D <-> ACD
    "k21": 1, "k22": 0,  # AC + B <-> ABC
    "k23": 1, "k24": 0,  # AB + C <-> ABC

    "k25": 1, "k26": 0,  # AB + CD <-> ABCD
    "k27": 1, "k28": 0,  # AC + BD <-> ABCD

    "k29": 1, "k30": 0,  # ABD + C <-> ABCD
    "k31": 1, "k32": 0,  # BCD + A <-> ABCD
    "k33": 1, "k34": 0,  # ACD + B <-> ABCD
    "k35": 1, "k36": 0   # ABC + D <-> ABCD
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

    {"reactants": {"BD":1, "C":1}, "products": {"BCD":1}, "k": "k13"},
    {"reactants": {"BCD":1},       "products": {"BD":1, "C":1}, "k": "k14"},

    {"reactants": {"B":1, "CD":1}, "products": {"BCD":1}, "k": "k15"},
    {"reactants": {"BCD":1},       "products": {"B":1, "CD":1}, "k": "k16"},

    {"reactants": {"CD":1, "A":1}, "products": {"ACD":1}, "k": "k17"},
    {"reactants": {"ACD":1},       "products": {"CD":1, "A":1}, "k": "k18"},

    {"reactants": {"AC":1, "D":1}, "products": {"ACD":1}, "k": "k19"},
    {"reactants": {"ACD":1},       "products": {"AC":1, "D":1}, "k": "k20"},

    {"reactants": {"AC":1, "B":1}, "products": {"ABC":1}, "k": "k21"},
    {"reactants": {"ABC":1},       "products": {"AC":1, "B":1}, "k": "k22"},

    {"reactants": {"AB":1, "C":1}, "products": {"ABC":1}, "k": "k23"},
    {"reactants": {"ABC":1},       "products": {"AB":1, "C":1}, "k": "k24"},

    # dimer + dimer <-> tetramer
    {"reactants": {"AB":1, "CD":1}, "products": {"ABCD":1}, "k": "k25"},
    {"reactants": {"ABCD":1},       "products": {"AB":1, "CD":1}, "k": "k26"},

    {"reactants": {"AC":1, "BD":1}, "products": {"ABCD":1}, "k": "k27"},
    {"reactants": {"ABCD":1},       "products": {"AC":1, "BD":1}, "k": "k28"},

    # trimer + monomer <-> tetramer (8 reactions)
    {"reactants": {"ABD":1, "C":1}, "products": {"ABCD":1}, "k": "k29"},
    {"reactants": {"ABCD":1},       "products": {"ABD":1, "C":1}, "k": "k30"},

    {"reactants": {"BCD":1, "A":1}, "products": {"ABCD":1}, "k": "k31"},
    {"reactants": {"ABCD":1},       "products": {"BCD":1, "A":1}, "k": "k32"},

    {"reactants": {"ACD":1, "B":1}, "products": {"ABCD":1}, "k": "k33"},
    {"reactants": {"ABCD":1},       "products": {"ACD":1, "B":1}, "k": "k34"},

    {"reactants": {"ABC":1, "D":1}, "products": {"ABCD":1}, "k": "k35"},
    {"reactants": {"ABCD":1},       "products": {"ABC":1, "D":1}, "k": "k36"},
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

def odes(t,y):
    """
    Return odes for the system.

    :param t: time
    :param y: state vector
    :return: odes
    """
    A, B, C, D, AB, AC, BD, CD, ABC, ABD, ACD, BCD, ABCD = y
    km = rates

    dA = (km["k2"]*AB + km["k4"]*AC + km["k12"]*ABD + km["k18"]*ACD + km["k32"]*ABCD
            - (km["k1"]*B + km["k3"]*C + km["k11"]*BD + km["k17"]*CD + km["k31"]*BCD)*A)

    dB = (km["k2"]*AB + km["k6"]*BD + km["k16"]*BCD + km["k22"]*ABC + km["k34"]*ABCD
            - (km["k1"]*A + km["k5"]*D + km["k15"]*CD + km["k21"]*AC + km["k33"]*ACD)*B)

    dC = (km["k4"]*AC + km["k8"]*CD + km["k14"]*BCD + km["k24"]*ABC + km["k30"]*ABCD
            - (km["k3"]*A + km["k7"]*D + km["k13"]*BD + km["k23"]*AB + km["k29"]*ABD)*C)

    dD = (km["k6"]*BD + km["k8"]*CD + km["k10"]*ABD + km["k20"]*ACD + km["k36"]*ABCD
            - (km["k5"]*B + km["k7"]*C + km["k9"]*AB + km["k19"]*AC + km["k35"]*ABC)*D)

    dAB = (km["k1"]*A*B + km["k10"]*ABD + km["k24"]*ABC + km["k26"]*ABCD
            - (km["k2"] + km["k9"]*D + km["k23"]*C + km["k25"]*CD)*AB)

    dAC = (km["k3"]*A*C + km["k20"]*ACD + km["k22"]*ABC + km["k28"]*ABCD
            - (km["k4"] + km["k19"]*D + km["k21"]*B + km["k27"]*BD)*AC)

    dBD = (km["k5"]*B*D + km["k12"]*ABD + km["k14"]*BCD + km["k28"]*ABCD
            - (km["k6"] + km["k11"]*A + km["k13"]*C + km["k27"]*AC)*BD)

    dCD = (km["k7"]*C*D + km["k16"]*BCD + km["k18"]*ACD + km["k26"]*ABCD
            - (km["k8"] + km["k15"]*B + km["k17"]*A + km["k25"]*AB)*CD)

    dABC = (km["k21"]*B*AC + km["k23"]*C*AB + km["k36"]*ABCD
                - (km["k22"] + km["k24"] + km["k35"]*D)*ABC)

    dABD = (km["k9"]*D*AB + km["k11"]*A*BD + km["k30"]*ABCD
                - (km["k10"] + km["k12"] + km["k29"]*C)*ABD)

    dACD = (km["k17"]*A*CD + km["k19"]*D*AC + km["k34"]*ABCD
                - (km["k18"] + km["k20"] + km["k33"]*B)*ACD)

    dBCD = (km["k13"]*C*BD + km["k15"]*B*CD + km["k32"]*ABCD
                - (km["k14"] + km["k16"] + km["k31"]*A)*BCD)

    dABCD = (km["k25"]*AB*CD + km["k27"]*AC*BD + km["k29"]*C*ABD + km["k31"]*A*BCD
                + km["k33"]*B*ACD + km["k35"]*D*ABC
                - (km["k26"] + km["k28"] + km["k30"] + km["k32"] + km["k34"] + km["k36"])*ABCD)

    return [dA, dB, dC, dD, dAB, dAC, dBD, dCD, dABC, dABD, dACD, dBCD, dABCD]

if __name__ == "__main__":
    # initial counts: set these as you like
    initial_counts = np.zeros(n_species, dtype=int)
    initial_counts[idx["A"]] = 100
    initial_counts[idx["B"]] = 100
    initial_counts[idx["C"]] = 100
    initial_counts[idx["D"]] = 100
    # all complexes start at 0

    duration = 500.0  # simulation time

    times, history = gillespie_ssa(initial_counts, duration, reactions,
                                   reactant_lists, stoich_changes, rates)

    
    # Solve deterministic ODEs for comparison
    y0 = initial_counts.astype(float)
    
    t_span = (0, duration) 
    t_eval = np.linspace(*t_span, 1000) 

    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA')

    # Plotting results
    for i,s in enumerate(species):
        plt.figure(figsize=(6,3))
        plt.plot(times, history[s], label=f"{s} (SSA)")
        plt.plot(sol.t, sol.y[i], '--', label=f"{s} (ODE)")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s} (SSA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"full_square_{s}.png", dpi=200)
        plt.close()

    print("SSA finished.")

    #### VISUALIZATION OF SPECIES PROPORTIONS ####

    #snapshot_times = np.linspace(0, duration, 3)
    snapshot_times = [0, 0.1, 0.3, 1, duration]

    # Categorize species by size
    monomers = [s for s in species if len(s) == 1]
    dimers   = [s for s in species if len(s) == 2]
    trimers  = [s for s in species if len(s) == 3]
    tetramers= [s for s in species if len(s) == 4]

    groups = {
        "Monomers": monomers,
        "Dimers": dimers,
        "Trimers": trimers,
        "Tetramers": tetramers,
    }

    # Helper to get snapshot index
    def nearest_index(array, value):
        return np.argmin(np.abs(array - value))

    for t_snap in snapshot_times:
        idx_snap = nearest_index(times, t_snap)
        state = {s: history[s][idx_snap] for s in species}
        total = sum(state.values())
        
        # Compute proportions per subspecies
        proportions = {g: np.array([state[s] / total for s in subspecies])
                    for g, subspecies in groups.items()}

        print(f"Proportions: {proportions["Tetramers"]}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(groups))

        # Consistent color set for species
        cmap = plt.get_cmap("tab20")
        color_map = {s: cmap(i % 20) for i, s in enumerate(species)}

        for s in species:
            # Find which group this species belongs to
            for j, (gname, subspecies) in enumerate(groups.items()):
                if s in subspecies:
                    frac = state[s] / total
                    ax.bar(gname, frac, bottom=bottom[j], color=color_map[s], label=s if bottom[j] == 0 else "")
                    bottom[j] += frac

        ax.set_ylabel("Proportion")
        ax.set_title(f"Species Proportions at t = {t_snap:.1f}")
        
        # Avoid duplicate entries in legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.savefig(f"snapshot_proportions_t{t_snap}.png", dpi=200)
        plt.close()



