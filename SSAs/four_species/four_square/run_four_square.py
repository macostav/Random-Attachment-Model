import numpy as np
import matplotlib.pyplot as plt
from four_square import (gillespie_ssa, species, idx,
 reactions, reactant_lists, stoich_changes, rates)

"""
Run the four-square simulation to allow easier plotting and analysis across multiple runs.
"""

if __name__ == "__main__":
    # Simulation parameters
    num_runs = 100         # number of independent simulations
    duration = 5           # simulation time
    t_eval = np.linspace(0, duration, 1000)  # time grid for aligned statistics

    # Initial conditions
    initial_counts = np.zeros(len(species), dtype=int)
    initial_counts[idx["A"]] = 100
    initial_counts[idx["B"]] = 100
    initial_counts[idx["C"]] = 100
    initial_counts[idx["D"]] = 100

    # Storage for all runs
    all_trajectories = {s: [] for s in species}

    # Times at which simulation ended
    end_times = []

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

        # Record when simulation ended
        end_times.append(times[-1])

        # Interpolate onto fixed time grid, i.e. make all x points the same across runs
        for s in species:
            y = np.interp(t_eval, times, history[s]) # !!! This will extrapolate after simulation ends; creating a flat line
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

    # Truncate the end times to the shortest simulation duration to avoid extrapolated data
    t_final = min(end_times)
    valid_indices = t_eval <= t_final

    ## PLOT MEAN +- SD ##
    print("Plotting ensemble statistics...")

    for s in species:
        mean, std = stats[s]

        plt.figure(figsize=(6,4))
        plt.errorbar(t_eval[valid_indices], mean[valid_indices], yerr = std[valid_indices], fmt = 'none', ecolor = "red", capsize = 1, label=f"SD")
        plt.plot(t_eval[valid_indices], mean[valid_indices], label=f"Mean", color="blue", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s} — {num_runs} runs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"ensemble_{s}_forward.png", dpi=200)
        plt.close()

    ## HISTOGRAM SHOWING SIMULATION DURATION ##
    plt.figure(figsize=(6,4))
    plt.hist(end_times, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Simulation end time")
    plt.ylabel("Number of runs")
    plt.title("Simulation durations")
    plt.tight_layout()
    plt.savefig("simulation_end_times_hist.png", dpi=200)
    plt.close()

    ## VISUALIZATION OF SPECIES PROPORTIONS ##

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
        return np.argmin(np.abs(array - value)) # Index where this is minimized

    # Colors
    cmap = plt.get_cmap("tab20")
    color_map = {s: cmap(i % 20) for i, s in enumerate(species)}

    print("Generating ensemble composition snapshots with species and group SDs...")

    snapshot_times = [0, 0.1, 0.3, 1, duration]

    # Species groups
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

    # Colors
    cmap = plt.get_cmap("tab20")
    color_map = {s: cmap(i % 20) for i, s in enumerate(species)}

    # TODO CHECK ALL THIS
    for t_snap in snapshot_times:
        idx_snap = np.argmin(np.abs(t_eval - t_snap))

        # Compute mean & SD for each species
        mean_state = {s: np.mean(all_trajectories[s][:, idx_snap]) for s in species}
        std_state  = {s: np.std(all_trajectories[s][:, idx_snap]) for s in species}

        total_mean = sum(mean_state[s] * len(s) for s in species)
        total_std = np.sqrt(sum(std_state[s]**2 * len(s)**2 for s in species))

        # Proportions for each species; careful not to divide by zero
        proportions_mean = {}
        proportions_std  = {}

        for s in species:
            # Mean proportion
            if total_mean > 0:
                p_mean = (mean_state[s] * len(s)) / total_mean
            else:
                p_mean = 0.0

            # If mean count is zero, the proportion is exactly zero with no uncertainty
            if mean_state[s] == 0 or total_mean == 0:
                p_std = 0.0

            else:
                # Compute relative variance only when safe
                rel_var = ((std_state[s] / mean_state[s])**2 +
                        (total_std   / total_mean)**2)

                p_std = p_mean * np.sqrt(rel_var)

            proportions_mean[s] = p_mean
            proportions_std[s]  = p_std

        # Prepare stacked barplot
        fig, ax = plt.subplots(figsize=(8,5))
        bottom = np.zeros(len(groups))

        for j, (gname, subspecies) in enumerate(groups.items()):
            # Compute group-level proportions across all runs for SD
            group_std  = np.sqrt(sum(proportions_std[s]**2 for s in subspecies))

            # Plot each species slice
            for s in subspecies:
                frac = proportions_mean[s]
                frac_std = proportions_std[s]

                ax.bar(gname, frac, bottom=bottom[j], color=color_map[s], label=s)

                # Species-level SD
                ax.errorbar(gname, bottom[j]+frac/2, yerr=frac_std,
                            fmt='none', ecolor='black', capsize=2, elinewidth=1)

                bottom[j] += frac

            # Group-level SD (red)
            group_top = sum(proportions_mean[s] for s in subspecies) # Identifies where the SD bar will go
            ax.errorbar(gname, group_top, yerr=group_std, fmt='none', ecolor='red', lw=2, capsize=4)

        ax.set_ylabel("Proportion of total molecules")
        ax.set_title(f"Ensemble Proportions at t = {t_snap:.2f} ({num_runs} runs)")

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05,1), loc="upper left")

        plt.tight_layout()
        plt.savefig(f"ensemble_snapshot_t{t_snap}_SD.png", dpi=200)
        plt.close()

        print(f" Proportion for ABCD: {proportions_mean["ABCD"]} +- {proportions_std["ABCD"]}") # !!! Get proportion of a species

    print("Ensemble snapshots with species & group SD complete.")
