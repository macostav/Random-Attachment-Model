# What is in this folder?

This folder contains a refactored, and updated version of the four-square simulation. In here, we ensure that our system obeys detailed balance by enforcing the condition that 

$$\frac{k_{\text{on}}}{k_{\text{off}}} = e^{-\frac{\Delta U}{K_B T}} \approx e^{-\Delta U},$$

assuming that $1/K_B T$ gets absorbed into the energy term. We achieve this by defining $k_{\text{on}} = C$ (some constant), and then $k_{\text{off}} = k_{\text{on}} e^{-\Delta U}$. To determine this constant, we note that the following applies for any reaction rate $k$:

$$
\begin{aligned}
k &\sim (\text{mean collision time})^{-1} \\
&\sim \biggl(\frac{L^2}{\sum_r D_r}\biggr)^{-1} = \frac{\sum_r D_r}{L^2},
\end{aligned}
$$

where $L$ is the length of the space where particles are moving, and $D_r$ represents the diffussion coefficient of species $r$. For the purposes of this simulation, we end up choosing 

$$D_r \sim \frac{1}{\text{mass}(r)} \sim \frac{1}{\sqrt{\text{\# particles}}}.$$

# How is this folder organized?

- `main.py` is where the simulation is run, and the results are plotted.
- `species.py` defines the species in the simulation, sets indices, and includes helper functions.
- `reactions.py` contains reaction dictionaries and stoichiometry information.
- `rates.py` calcualtes kon/koff based on diffusion and interaction energies.
- `ssa.py` has the Gillespie SSA.
- `odes.py` contains the deterministic ODEs describing the macroscopic behaviour of the system.
- `plot_utils.py` has helper functions for plotting.
- `config.py` has general parameters for the simulation.
