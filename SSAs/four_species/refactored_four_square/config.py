# Simulation parameters
DUMMY_L2 = 1.0                 # L^2 for diffusion-based rates
SIM_DURATION = 100.0            # total simulation time
MAX_STEPS = int(1e7)            # max SSA steps

# Initial counts of monomers
INITIAL_COUNTS = {
    "A": 50,
    "B": 50,
    "C": 50,
    "D": 50
    # all complexes start at 0
}

# Energies for allowed bonds
BOND_ENERGY = {
    ("A","B"): 1.0,
    ("A","C"): 1.0,
    ("B","D"): 1.0,
    ("C","D"): 1.0
}

# Plotting options
PLOT_DPI = 200
COLORMAP = "tab20"
SNAPSHOT_TIMES = [0, 0.1, 0.3, 1, SIM_DURATION]