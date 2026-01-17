import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_DPI, SNAPSHOT_TIMES, COLORMAP

def plot_species_trajectory(times, history, species, sol=None, filename_prefix="species_trajectory"):
    for i, s in enumerate(species):
        plt.figure(figsize=(6,3))
        plt.plot(times, history[s], label=f"{s} (SSA)")
        if sol is not None:
            plt.plot(sol.t, sol.y[i], '--', label=f"{s} (ODE)")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{s}.png", dpi=PLOT_DPI)
        plt.close()

def plot_species_snapshots(times, history, species, snapshot_times=None):
    if snapshot_times is None:
        snapshot_times = SNAPSHOT_TIMES

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

    cmap = plt.get_cmap(COLORMAP)
    color_map = {s: cmap(i % 20) for i, s in enumerate(species)}

    def nearest_index(array, value):
        return np.argmin(np.abs(array - value))

    for t_snap in snapshot_times:
        idx_snap = nearest_index(times, t_snap)
        state = {s: history[s][idx_snap] for s in species}
        total = sum(state[s] * len(s) for s in species)

        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(groups))

        for s in species:
            for j, (gname, subspecies) in enumerate(groups.items()):
                if s in subspecies:
                    frac = (state[s] * len(s)) / total
                    ax.bar(gname, frac, bottom=bottom[j], color=color_map[s], label=s)
                    bottom[j] += frac

        ax.set_ylabel("Proportion of total molecules")
        ax.set_title(f"Species Proportions at t = {t_snap:.1f}")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(f"snapshot_proportions_t{t_snap}.png", dpi=PLOT_DPI)
        plt.close()