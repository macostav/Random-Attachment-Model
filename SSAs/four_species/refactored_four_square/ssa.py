import numpy as np
import random, math

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

def gillespie_ssa_with_log(initial_counts, t_max, species, reactions, reactant_lists, stoich_changes, rates,
                        max_steps=int(1e7)):
    counts = np.array(initial_counts, dtype=int)
    t = 0.0

    history = {s: [counts[idx_s]] for idx_s, s in enumerate(species)}
    times = [t]
    events = []  # list of dicts: {"t":..., "ri":..., "name":..., "counts": array}

    for step in range(max_steps):
        a = compute_propensities(counts, reactant_lists, rates, reactions)
        a0 = a.sum()
        if a0 <= 0.0:
            # record final state and break
            events.append({"t": t, "ri": None, "name": "STOP_no_propensity", "counts": counts.copy()})
            break

        r1 = random.random()
        r2 = random.random()
        tau = -math.log(r1) / a0
        t += tau
        if t > t_max:
            # stop (we do not apply the reaction that would pass t_max)
            events.append({"t": t_max, "ri": None, "name": "STOP_tmax_reached", "counts": counts.copy()})
            # append final time to history
            times.append(t_max)
            for idx_s, s in enumerate(species):
                history[s].append(int(counts[idx_s]))
            break

        # choose reaction
        cum = np.cumsum(a)
        target = r2 * a0
        ri = np.searchsorted(cum, target)
        # apply reaction stoichiometry
        counts += stoich_changes[ri]

        # record event
        events.append({
            "t": t,
            "ri": ri,
            "name": f"r{ri+1}_{reactions[ri]['k']}",
            "counts": counts.copy()
        })

        # save history
        times.append(t)
        for idx_s, s in enumerate(species):
            history[s].append(int(counts[idx_s]))

    return np.array(times), history, events