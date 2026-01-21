from species import species, n_particles
from reactions import reactions
from config import BOND_ENERGY, DUMMY_L2
import numpy as np

def compute_rates(reactants, products, bond_energy):
    """Compute forward and backward rates based on diffusion + bond energy"""
    Dsum = sum(1/np.sqrt(n_particles(s)) for s in reactants)
    kon = Dsum / DUMMY_L2
    # Sum bond energies formed in this reaction
    species_list = list(products.keys())[0]
    delta_U = 0
    for i in range(len(species_list)):
        for j in range(i+1, len(species_list)):
            pair = tuple(sorted((species_list[i], species_list[j])))
            if pair in bond_energy:
                delta_U += bond_energy[pair]
    koff = kon * np.exp(delta_U) # TODO: check whether +- is correct here.
    return kon, koff

rates = {}
for i, r in enumerate(reactions[::2]):  # forward reactions
    kf, kb = compute_rates(r['reactants'], r['products'], BOND_ENERGY)
    rates[r['k']] = kf
    rb = reactions[2*i+1]  # backward reaction
    rates[rb['k']] = kb