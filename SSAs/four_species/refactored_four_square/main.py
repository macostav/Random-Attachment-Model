import numpy as np
from collections import defaultdict
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from species import species, idx
from reactions import reactions, reactant_lists, stoich_changes
from rates import rates
from ssa import gillespie_ssa_with_log
from odes import odes
from config import INITIAL_COUNTS, SIM_DURATION
from plot_utils import plot_species_trajectory, plot_species_snapshots

# Initial counts array
initial_counts = np.zeros(len(species), dtype=int)
for s, n in INITIAL_COUNTS.items():
    initial_counts[idx[s]] = n

# Run SSA
times, history, events = gillespie_ssa_with_log(
    initial_counts, SIM_DURATION, species,
    reactions, reactant_lists, stoich_changes, rates
)

# Solve ODEs
y0 = initial_counts.astype(float)
t_span = (0, SIM_DURATION)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA')

# Plot
plot_species_trajectory(times, history, species, sol)
plot_species_snapshots(times, history, species)

