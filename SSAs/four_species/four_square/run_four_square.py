import numpy as np
import matplotlib.pyplot as plt
from four_square import (gillespie_ssa, species, idx,
 reactions, reactant_lists, stoich_changes, rates)

"""
Run the four-square simulation to allow easier plotting and analysis across multiple runs.
"""

if __name__ == "__main__":
    # Simulation parameters
    num_runs = 100             # number of independent stochastic simulations
    duration = 50.0           # simulation time
    t_eval = np.linspace(0, duration, 100)  # time grid for aligned statistics

    # Initial conditions
    initial_counts = np.zeros(len(species), dtype=int)
    initial_counts[idx["A"]] = 100
    initial_counts[idx["B"]] = 100
    initial_counts[idx["C"]] = 100
    initial_counts[idx["D"]] = 100

    # Storage for all runs
    all_trajectories = {s: [] for s in species}

    # Run simulations
    print(f"Running {num_runs} SSA simulations...")

    for r in range(num_runs):
        times, history = gillespie_ssa(
            initial_counts,
            duration,
            reactions,
            reactant_lists,
            stoich_changes,
            rates
        )

        # Interpolate onto fixed time grid, i.e. make all x points the same across runs
        for s in species:
            y = np.interp(t_eval, times, history[s])
            all_trajectories[s].append(y)

    # Convert lists to arrays
    for s in species:
        all_trajectories[s] = np.array(all_trajectories[s])  # shape (num_runs, len(t_eval))

    # Compute mean and SD; create array of means at each time, and array of SDs at each time
    stats = {}
    for s in species:
        mean = np.mean(all_trajectories[s], axis=0)
        std  = np.std(all_trajectories[s], axis=0)
        stats[s] = (mean, std)

    # Plot mean +- SD
    print("Plotting ensemble statistics...")

    for s in species:
        mean, std = stats[s]

        plt.figure(figsize=(6,4))
        plt.errorbar(t_eval, mean, yerr = std, fmt = 'none', ecolor = "red", capsize = 1, label=f"SD")
        plt.plot(t_eval, mean, label=f"Mean", color="blue", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s} — {num_runs} runs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"ensemble_{s}_forward.png", dpi=200)
        plt.close()

    print("All done.")