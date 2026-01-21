# Simulation parameters
DUMMY_L2 = 1.0                 # L^2 for diffusion-based rates
SIM_DURATION = 3000000.0            # total simulation time
MAX_STEPS = int(1e7)            # max SSA steps

# Initial counts of monomers
INITIAL_COUNTS = {
    "A": 200,
    "B": 200,
    "C": 200,
    "D": 200
    # all complexes start at 0
}

# Energies for allowed bonds
BOND_ENERGY = {
    ("A","B"): -1.0,
    ("A","C"): -2.0,
    ("B","D"): -5.0,
    ("C","D"): -8.0
}

# Plotting options
PLOT_DPI = 200
COLORMAP = "tab20"
SNAPSHOT_TIMES = [0, 0.1, 0.3, 1, SIM_DURATION]