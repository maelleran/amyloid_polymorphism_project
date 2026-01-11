#physical constants
N_A = 6.022e23

#constants of the system
m_tot = 1e-6
V = 1e-12

# reaction orders for A
n_2A = 2
n_cA = 2

# Kinetic rates for A
k_nA = 1e-4
k_plusA = 1e6
k_2A = 1e4
alpha1A = k_nA*(m_tot)**n_cA*N_A*V
alpha2A = k_2A*(m_tot)**n_2A
muA = 2*k_plusA*m_tot

# reaction orders for B
n_2B = 2
n_cB = 2

# kinetic rates for B
k_nB = 1e-4
k_plusB = 1e6
k_2B = 1e4
alpha1B = k_nB*(m_tot)**n_cA*N_A*V
alpha2B = k_2B*(m_tot)**n_2A
muB = 2*k_plusB*m_tot

# time parameters
t_max = 35000
n_switch = 24
tau_leap =  1 / (2*max(k_plusB, k_plusA) * m_tot* n_switch)
