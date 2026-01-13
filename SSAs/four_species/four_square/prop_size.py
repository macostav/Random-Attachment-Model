import numpy as np
import matplotlib.pyplot as plt
from four_square import (gillespie_ssa, species, idx,
 reactions, reactant_lists, stoich_changes, rates)

"""
In the all-forward experiment, check how the system size affects the proportion of each species.
"""

if __name__ == "__main__":
    num_runs = 100
    duration = 5
    t_eval = np.linspace(0, duration, 1000)
    system_sizes = [40, 100, 200, 400, 800, 1000, 2000, 4000, 10000]

    # Categorize species by size
    monomers  = [s for s in species if len(s) == 1]
    dimers    = [s for s in species if len(s) == 2]
    trimers   = [s for s in species if len(s) == 3]
    tetramers = [s for s in species if len(s) == 4]

    groups = {
        "Monomers": monomers,
        "Dimers": dimers,
        "Trimers": trimers,
        "Tetramers": tetramers
    }

    # Storage
    group_fractions = {g: [] for g in groups}
    group_std       = {g: [] for g in groups}

    for N_total_system in system_sizes:
        print(f"\nSystem size {N_total_system}:")

        # Initialize monomers
        initial_counts = np.zeros(len(species), dtype=int)
        count_per_monomer = N_total_system // 4
        for s in monomers:
            initial_counts[idx[s]] = count_per_monomer

        # Run simulations
        all_trajectories = {s: [] for s in species}
        for r in range(num_runs):
            times, history = gillespie_ssa(
                initial_counts,
                duration,
                reactions,
                reactant_lists,
                stoich_changes,
                rates
            )
            for s in species:
                y = np.interp(t_eval, times, history[s])
                all_trajectories[s].append(y)

        # Convert to arrays
        for s in species:
            all_trajectories[s] = np.array(all_trajectories[s])

        # Final snapshot
        idx_final = -1
        mean_state   = {s: np.mean(all_trajectories[s][:, idx_final]) for s in species}
        std_state    = {s: np.std(all_trajectories[s][:, idx_final], ddof=1) for s in species}
        stderr_state = {s: std_state[s] / np.sqrt(num_runs) for s in species}

        # Total particles
        N_total_final = sum(mean_state[s] * len(s) for s in species)

        # Species fractions
        proportions_mean = {}
        proportions_std  = {}
        for s in species:
            weight = len(s)
            p_mean = (mean_state[s] * weight) / N_total_final
            p_std  = (stderr_state[s] * weight) / N_total_final if mean_state[s] > 0 else 0.0
            proportions_mean[s] = p_mean
            proportions_std[s]  = p_std

        # ---- Sum by group ----
        for gname, subspecies in groups.items():
            group_frac = sum(proportions_mean[s] for s in subspecies)
            group_sigma = np.sqrt(sum(proportions_std[s]**2 for s in subspecies))

            group_fractions[gname].append(group_frac)
            group_std[gname].append(group_sigma)

        # Print results
        print("Group fractions:")
        for gname in groups:
            print(f"{gname}: {group_fractions[gname][-1]:.4f} ± {group_std[gname][-1]:.4f}")

    # ---- Plot final fractions by group ----
    plt.figure(figsize=(8,5))
    for gname in groups:
        plt.errorbar(
            system_sizes,
            group_fractions[gname],
            yerr=group_std[gname],
            marker='o',
            capsize=4,
            label=gname
        )
    plt.xlabel("Total system size (particles)")
    plt.ylabel("Final Proportion")
    plt.title("Final proportions of group vs system size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()