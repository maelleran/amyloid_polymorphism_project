import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde
import constants as C

def gillespie_algo():

    #Initial state  
    n, m, t = 0, 0, 0 
    m0 = int(C.m_tot*C.N_A*C.V)
    print(m0)
    #storage
    times = [t]
    m_list = [m]
    n_list = [n]

    while t < C.t_max:  
        #update reaction coefficients
        M_free = m0 - m 
        c_free = M_free / (C.N_A * C.V)
        if M_free <= 0: 
            break

        a1 = C.k_n*(c_free)**C.n_c*C.N_A*C.V #first nucleation
        a3 = C.k_2*(c_free)**C.n_2*m #second nucleation
        a2 = 2*C.k_plus*c_free*n #elongation
        a0 = a1 + a2 + a3
        if a0 <= 0:
            break  

        #random time 
    
        r_1 = random.random() 
        tau = (1/a0)*np.log(1/r_1)
        if tau < C.t_max: 
            t += tau
            #random reaction
            r_2 = random.random() 
            threshold = r_2*a0

            if threshold < a1 and M_free >= C.n_c: 
                n += 1
                m += C.n_c 
            elif threshold < a1 + a2 and M_free >= 1: 
                m += 1
            elif M_free >= C.n_2: 
                m += C.n_2
                n += 1

        times.append(t)
        m_list.append(m)
        n_list.append(n)

    return np.array(times), np.array(n_list), np.array(m_list)


