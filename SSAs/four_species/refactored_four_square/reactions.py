from species import idx
import numpy as np

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
stoich_changes = np.zeros((n_reactions, len(idx)), dtype=int)
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
