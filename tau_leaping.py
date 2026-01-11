import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde
import constants as C

def gillespie_algo():

    #Initial state  
    n, m, t = 0, 0, 0
    
    #storage
    times = [t]
    m_list = [m]
    n_list = [n]

    alpha1 = C.k_n * (C.m_tot ** C.n_c) * C.N_A * C.V
    mu     = 2 * C.k_plus * C.m_tot
    alpha2 = C.k_2 * (C.m_tot ** C.n_2)

    while t < C.t_max:  
        #update reaction coefficients
        if n < C.n_switch:
            a1 = alpha1
            a2 = mu * n
            a3 = alpha2 * m        	
            a0 = a1 + a2 + a3
            if a0 <= 0:
            	break  

        	#random time 
            r_1 = random.random() 
            tau = (1/a0)*np.log(1/r_1)
            t += tau
                
            r_2 = random.random() 
            threshold = r_2*a0

            if threshold < a1: 
                n += 1
                m += C.n_c 
            elif threshold < a1 + a2: 
                m += 1
            else: 
                m += C.n_2
                n += 1
        else:
            tau = C.tau_leap

            k1 = np.random.poisson(alpha1 * tau) 
            k2 = np.random.poisson(alpha2 * m * tau)  
            k3 = np.random.poisson(mu* n * tau) 

            # update state
            n += k1 + k2
            m += k1 * C.n_c +  k2 * C.n_2 + k3
            t += tau
            
        times.append(t)
        m_list.append(m)
        n_list.append(n)
    return np.array(times), np.array(n_list), np.array(m_list)

