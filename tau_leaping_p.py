import numpy as np
import random
import constants_p as C


def gillespie_algo():

    # Initial state
    nA = mA = 0
    nB = mB = 0
    t = 0.0

    first_nuclA = 0
    first_nuclB = 0

    # Storage
    times = [t]
    nA_list = [nA]
    mA_list = [mA]
    nB_list = [nB]
    mB_list = [mB]

    while t < C.t_max:
        
        if nA < C.n_switch or nB < C.n_switch:

            a1 = C.alpha1A
            a2 = C.muA * nA
            a3 = C.alpha2A * mA
            a4 = C.alpha1B
            a5 = C.muB * nB
            a6 = C.alpha2B * mB

            a0 = a1 + a2 + a3 + a4 + a5 + a6

            if a0 == 0:
                break

            r1 = random.random()
            tau = (1/a0)*np.log(1/r1)
            t += tau
            r2 = random.random()
            threshold = r2*a0

            if threshold < a1: 
                nA += 1
                mA += C.n_cA   
                first_nuclA += 1
            elif threshold < a1 + a2: 
                mA += 1
            elif threshold < a1 + a2 + a3:
                mA += C.n_2A
                nA += 1
            elif threshold < a1 + a2 + a3 + a4:
                nB += 1
                mB += C.n_cB 
                first_nuclB += 1
            elif threshold < a1 + a2 + a3 + a4 + a5:
                mB += 1
            else: 
                mB += C.n_2B
                nB += 1
                
        else:
            tau = C.tau_leap

            # Species A
            p1A = np.random.poisson(C.alpha1A * tau)
            p2A = np.random.poisson(C.alpha2A * mA * tau)
            p3A = np.random.poisson(C.muA * nA * tau)

            nA += p1A + p2A
            mA += C.n_cA * p1A + C.n_2A * p2A + p3A

            # Species B
            p1B = np.random.poisson(C.alpha1B * tau)
            p2B = np.random.poisson(C.alpha2B * mB * tau)
            p3B = np.random.poisson(C.muB * nB * tau)

            nB += p1B + p2B
            mB += C.n_cB * p1B + C.n_2B * p2B + p3B

            t += tau

        # Store
        times.append(t)
        nA_list.append(nA)
        mA_list.append(mA)
        nB_list.append(nB)
        mB_list.append(mB)

    return (
        np.array(times),
        np.array(nA_list),
        np.array(mA_list),
        np.array(nB_list),
        np.array(mB_list),
        first_nuclA,
        first_nuclB,
    )
