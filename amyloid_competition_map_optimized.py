import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from tqdm import tqdm  # optional

# Global parameters
n_c = 2 
n_2 = 2
m_tot = 1e-5

# Type A
k_plusA = 1e6
k_nA = 1e-4
k_2A = 1e4 
# Type B
k_nB = 1e-4 

# Initial Conditions
P01 = 0
M01 = 0
P02 = 0
M02 = 0
m0 = 1
 
def system(t, y, k_nA, k_2A, k_plusA, k_nB, k_2B, k_plusB, n_c, n_2):
    alpha11 = k_nA/k_plusA
    alpha21 = k_2A/k_plusA
    alpha12 = k_nB/k_plusA
    alpha22 = k_2B/k_plusA
    beta = k_plusB/k_plusA

    P1, M1, P2, M2, m = y

    dP1dt = alpha11*m**n_c + alpha21*m_tot*m**n_2*M1
    dM1dt = 2*m*P1
    dP2dt = alpha12*m**n_c + alpha22*m_tot*m**n_2*M2
    dM2dt = 2*m*P2*beta
    dmdt  = -2*m*P1-2*m*beta*P2

    return [dP1dt, dM1dt, dP2dt, dM2dt, dmdt]

def run_simulation(ratio_plus, ratio_2):
    k_plusB = k_plusA /ratio_plus 
    k_2B = k_2A / ratio_2
    sol = solve_ivp(system, [0, 60000], [P01, M01, P02, M02, m0],
                    method='LSODA', max_step=1000,
                    args=(k_nA, k_2A, k_plusA, k_nB, k_2B, k_plusB, n_c, n_2))
    M1_final = sol.y[1, -1]
    M2_final = sol.y[3, -1]

    if M1_final == 0: 
        return np.nan
    return M1_final/(M1_final + M2_final)

# Ratios and Grid 
ratios_plus = np.linspace(0.1, 10, 200)  # from 0.1 to 10
ratios_2 = np.linspace(0.1, 10, 200)
Rplus, R2 = np.meshgrid(ratios_plus, ratios_2, indexing='ij')

# --- PARALLELIZED SECTION (replace your nested loop or vectorize here) ---
def compute_row(i):
    """Compute one row of the grid for a fixed ratio_plus."""
    row = np.zeros(len(ratios_2))
    for j, r2 in enumerate(ratios_2):
        row[j] = run_simulation(ratios_plus[i], r2)
    return row

from tqdm import tqdm  # optional
grid_rows = Parallel(n_jobs=-1, backend='loky')(
    delayed(compute_row)(i) for i in tqdm(range(len(ratios_plus)))
)
grid = np.array(grid_rows)
# --- END PARALLELIZED SECTION ---

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(grid.T, origin='lower', aspect='auto',
           extent=[ratios_plus[0], ratios_plus[-1], ratios_2[0], ratios_2[-1]],
           cmap='inferno')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k_{+A}/k_{+B}$')
plt.ylabel(r'$k_{2A}/k_{2B}$')
plt.colorbar()
plt.tight_layout()
plt.show()


