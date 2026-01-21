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

# Reaction pairs rates
reversible_pairs = [
    ("k1","k2"), ("k3","k4"), ("k5","k6"), ("k7","k8"),
    ("k9","k10"), ("k11","k12"), ("k13","k14"), ("k15","k16"),
    ("k17","k18"), ("k19","k20"), ("k21","k22"), ("k23","k24"),
    ("k25","k26"), ("k27","k28"),
    ("k29","k30"), ("k31","k32"), ("k33","k34"), ("k35","k36")
]

def compute_net_fluxes(history, reactions, rates, species, window=1000):
    """
    Compute net flux k_fw <alpha> - k_bw <beta> using equilibrium averages.
    """
    equil_counts = {s: np.mean(history[s][-window:]) for s in species}

    net_fluxes = {}

    for kf, kb in reversible_pairs:
        rxn_fw = next(r for r in reactions if r["k"] == kf)
        rxn_bw = next(r for r in reactions if r["k"] == kb)

        k_fw = rates[kf]
        k_bw = rates[kb]

        c_alpha = np.prod([
            equil_counts[s]**n for s, n in rxn_fw["reactants"].items()
        ]) if rxn_fw["reactants"] else 1.0

        c_beta = np.prod([
            equil_counts[s]**n for s, n in rxn_bw["reactants"].items()
        ]) if rxn_bw["reactants"] else 1.0

        net_fluxes[f"{kf}/{kb}"] = k_fw * c_alpha - k_bw * c_beta

    return net_fluxes

def plot_net_flux_bars(flux_matrix, pair_labels, durations):
    n_durations, n_pairs = flux_matrix.shape
    x = np.arange(n_pairs)
    width = 0.8 / n_durations

    plt.figure(figsize=(14, 5))

    for i, T in enumerate(durations):
        plt.bar(
            x + i*width,
            flux_matrix[i],
            width,
            label=f"T = {T}"
        )

    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xticks(x + width*(n_durations-1)/2, pair_labels, rotation=45)
    plt.ylabel("Net Flux")
    plt.title("Detailed balance convergence vs simulation duration")
    plt.legend()
    plt.tight_layout()
    plt.show()

### SIMULATION CHECK ###

# Initial counts array
initial_counts = np.zeros(len(species), dtype=int)
for s, n in INITIAL_COUNTS.items():
    initial_counts[idx[s]] = n
"""
SIM_DURATIONS = [0.1, 1, 10, 100, 1000]
flux_by_duration = {}

for T in SIM_DURATIONS:
    times, history, events = gillespie_ssa_with_log(
        initial_counts, T, species,
        reactions, reactant_lists, stoich_changes, rates
    )

    flux_by_duration[T] = compute_net_fluxes(
        history, reactions, rates, species, window=500
    )

pair_labels = list(next(iter(flux_by_duration.values())).keys())

flux_matrix = np.array([
    [flux_by_duration[T][p] for p in pair_labels]
    for T in SIM_DURATIONS
])

plot_net_flux_bars(flux_matrix, pair_labels, SIM_DURATIONS)
"""
### ODE CHECK ###

# Solve ODEs
y0 = initial_counts.astype(float)
t_span = (0, SIM_DURATION)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method='LSODA')

c_star = sol.y[:, -1] # Equilibrium counts from ODEs
c_eq = {s: c_star[idx[s]] for s in species}

assert np.linalg.norm(sol.y[:, -1] - sol.y[:, -10]) < 1e-6 # Check that we have converged so dc/dt ~ 0

print(f"{'Reaction pair':<10} | {'Forward':>12} | {'Backward':>13} | {'Difference':>10}")
print("-"*55)

fw_fluxes = []
bw_fluxes = []
labels = []

for kf, kb in reversible_pairs:
    rxn_fw = next(r for r in reactions if r["k"] == kf)
    rxn_bw = next(r for r in reactions if r["k"] == kb)

    k_fw = rates[kf]
    k_bw = rates[kb]

    c_alpha = np.prod([
        c_eq[s]**n for s, n in rxn_fw["reactants"].items()
    ]) if rxn_fw["reactants"] else 1.0

    c_beta = np.prod([
        c_eq[s]**n for s, n in rxn_bw["reactants"].items()
    ]) if rxn_bw["reactants"] else 1.0

    flux_fw = k_fw * c_alpha
    flux_bw = k_bw * c_beta
    
    fw_fluxes.append(flux_fw)
    bw_fluxes.append(flux_bw)
    labels.append(f"{kf}/{kb}")

    print(f"{kf}/{kb:<6} | {flux_fw:12.6e} | {flux_bw:13.6e} | {(flux_fw-flux_bw):10.2e}")

fw_fluxes = np.array(fw_fluxes)
bw_fluxes = np.array(bw_fluxes)

# Plot
plt.figure(figsize=(6, 6))

plt.scatter(
    fw_fluxes,
    bw_fluxes,
    s=60,
    alpha=0.8
)

# Diagonal y = x (perfect detailed balance)
min_val = min(fw_fluxes.min(), bw_fluxes.min())
max_val = max(fw_fluxes.max(), bw_fluxes.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    '--',
    linewidth=2,
    label="Detailed balance (fw = bw)"
)

plt.xlabel("Forward flux")
plt.ylabel("Backward flux")
plt.title("Detailed Balance Check (ODE at Equilibrium)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.show()
