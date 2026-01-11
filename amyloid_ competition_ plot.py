import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# Global parameters
n_c = 2 
n_2 = 2
m_tot = 1e-5

# Type A
k_plusA = 1e6
k_nA = 1e-4
k_2A = 1e4
# Type B
k_plusB = 2e6
k_nB = 1e-4
k_2B = 1e4


# Initial Conditions
P01 = 0
M01 = 0
P02 = 0
M02 = 0
m0 = 1

# system of equations 
def system2(t, y, k_nA, k_2A, k_plusA, k_nB, k_2B, k_plusB, n_c, n_2):
    alpha11= k_nA/k_plusA
    alpha21 = k_2A*m_tot/k_plusA
    alpha12 = k_nB/k_plusA
    alpha22 = k_2B*m_tot/k_plusA
    beta = k_plusB/k_plusA

    P1, M1, P2, M2, m = y

    dP1dt = alpha11*m**n_c + alpha21*m**n_2*M1
    dM1dt = 2*m*P1
    dP2dt = alpha12*m**n_c + alpha22*m**n_2*M2
    dM2dt = 2*m*P2*beta
    dmdt  = -2*m*P1-2*m*beta*P2

    return [dP1dt, dM1dt, dP2dt, dM2dt, dmdt]

#resolution 
t_max = 60000
sol = solve_ivp(system2, [0, t_max], [P01, M01, P02, M02, m0], method='RK45', args=(k_nA, k_2A, k_plusA, k_nB, k_2B, k_plusB, n_c, n_2), max_step= 1000)
M1 = sol.y[1]
M2 = sol.y[3]
for i in range(len(M1)):
    print(M1[i], M2[i])

plt.figure(figsize=(8,5))
plt.plot(sol.t/600, M1, label="Type A")
plt.plot(sol.t/600, M2, label="Type B")

plt.xlabel('Time (min)')
plt.ylabel('Fractional Fibril Mass')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#petit problème de time scale 