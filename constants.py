import numpy as np

# Physical constants 
N_A = 6.022e23

# System parameters 
m_tot = 1e-6 #70microM (43microM of A, 64 microM of B)
V = 1e-12 # V=2mL? 
N_A = 6.022e23

# Kinetic rates 
k_n = 1e-4
k_plus = 1e6
k_2 = 1e4

# Reaction orders
n_2 = 2
n_c = 2 

# time parameters
t_max = 30000
n_switch = 24
tau_leap = 1 / (2* k_plus * m_tot* n_switch)
