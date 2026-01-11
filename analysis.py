import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import constants_p as C
from tau_leaping_p import gillespie_algo


number_trajectories = 5000
t_grid = np.linspace(0, C.t_max, 500)

single = True
many = False
mean = True
dev = True
distribution = True

if single == True: 

    times, nA_traj, mA_traj, nB_traj, mB_traj, first_nuclA, first_nuclB = gillespie_algo()

    plt.figure(figsize=(8,4))
    plt.plot(times, mA_traj, label="Mass A")
    plt.plot(times, mB_traj, label="Mass B")
    plt.xlabel("time")
    plt.ylabel("aggregate mass")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

all_times = []
all_massesA = []
all_fibrilsA = []
all_massesB = []
all_fibrilsB = []
all_nuclA = []
all_nuclB = []

for k in range(number_trajectories):
    t, nA_traj, mA_traj, nB_traj, mB_traj, first_nuclA, first_nuclB = gillespie_algo()

    all_times.append(t)
    all_massesA.append(mA_traj)
    all_fibrilsA.append(nA_traj)
    all_massesB.append(mB_traj)
    all_fibrilsB.append(nB_traj)
    all_nuclA.append(first_nuclA)
    all_nuclB.append(first_nuclB)
    if k%100 == 0:
        print(k)

def nucleation_events(all_nucleationA, all_nucleationB): 
    lenghtA = len(all_nucleationA)
    lenghtB = len(all_nucleationB)
    meanA = 0
    meanB = 0
    for nucl in all_nucleationA: 
        meanA += nucl
    for nucl in all_nucleationB: 
        meanB += nucl
    return meanA/lenghtA, meanB/lenghtB

if many == True: 
    
    plt.figure(figsize=(10, 5))

    for t, m in zip(all_times, all_massesA):
        plt.plot(t/3600, m*1e6 / (C.N_A * C.V))
    for t, m in zip(all_times, all_massesB):
        plt.plot(t/3000, m*1e6 / (C.N_A * C.V))
    plt.xlabel("time (h)")
    plt.ylabel("Fibril mass concentration (µM)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    n_fibrils = False

    if n_fibrils == True:

        plt.figure(figsize=(10, 5))

        for t, n in zip(all_times, all_fibrilsA):
            plt.plot(t/3600, n)
        for t, n in zip(all_times, all_fibrilsB):
            plt.plot(t/3600, n)

        plt.xlabel("time")
        plt.ylabel("number of fibrils n")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

if mean == True: 
    # INTERPOLATION AND MEAN A
    t_grid = np.linspace(0, C.t_max, 500)
    interp_massesA = []

    for t, mA in zip(all_times, all_massesA):
        interp_mA = np.interp(t_grid, t, mA)
        interp_massesA.append(interp_mA)
    interp_massesA = np.array(interp_massesA)  
    mean_massA = np.mean(interp_massesA, axis=0)

    # INTERPOLATION AND MEAN B
    interp_massesB = []

    for t, mB in zip(all_times, all_massesB):
        interp_mB = np.interp(t_grid, t, mB)
        interp_massesB.append(interp_mB)
    interp_massesB = np.array(interp_massesB)  
    mean_massB = np.mean(interp_massesB, axis=0)

    plt.figure(figsize=(10, 5))
    for i in range(number_trajectories):
        plt.plot(t_grid/3600, interp_massesA[i]*1e6 / (C.N_A * C.V), color="gray", alpha=0.2)
    plt.plot(t_grid/3600, mean_massA*1e6 / (C.N_A * C.V), color="black", label="Mean A", linewidth=2)

    for i in range(number_trajectories):
        plt.plot(t_grid/3600, interp_massesB[i]*1e6 / (C.N_A * C.V), color="gray", alpha=0.2)
    plt.plot(t_grid/3600, mean_massB*1e6 / (C.N_A * C.V), color="blue", label="Mean B", linewidth=2)

    plt.xlabel("time (h)")
    plt.ylabel("Fibril mass concentration (µM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if dev == True:

    t_grid = np.linspace(0, C.t_max, 500)

    # MEDIAN AND QUARTILES A
    median_massA = np.median(interp_massesA, axis=0)
    q25_A = np.percentile(interp_massesA, 25, axis=0)
    q75_A = np.percentile(interp_massesA, 75, axis=0)

    # MEDIAN AND QUARTILES B
    median_massB = np.median(interp_massesB, axis=0)
    q25_B = np.percentile(interp_massesB, 25, axis=0)
    q75_B = np.percentile(interp_massesB, 75, axis=0)
     
    plt.figure(figsize=(10, 5))
    plt.fill_between(t_grid/3600, q25_A*1e6 / (C.N_A * C.V), q75_A*1e6 / (C.N_A * C.V), alpha=0.3, label="25–75% quantile")
    plt.plot(t_grid/3600, median_massA*1e6 / (C.N_A * C.V), color="black", label="Median", linewidth=2)
    plt.fill_between(t_grid/3600, q25_B*1e6 / (C.N_A * C.V), q75_B*1e6 / (C.N_A * C.V), alpha=0.3, label="25–75% quantile")
    plt.plot(t_grid/3600, median_massB*1e6 / (C.N_A * C.V), color="blue", label="Median", linewidth=2)
    plt.xlabel("time (h)")
    plt.ylabel("Fibril mass concentration (µM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # FIBRILS A
    interp_fibrilsA = []
    for t, nA in zip(all_times, all_fibrilsA):
        interp_fA = np.interp(t_grid, t, nA)
        interp_fibrilsA.append(interp_fA)
    interp_fibrilsA = np.array(interp_fibrilsA)
    mean_fibrilsA = np.mean(interp_fibrilsA, axis=0)

    # FIBRILS B
    interp_fibrilsB = []
    for t, nB in zip(all_times, all_fibrilsB):
        interp_fB = np.interp(t_grid, t, nB)
        interp_fibrilsB.append(interp_fB)
    interp_fibrilsB = np.array(interp_fibrilsB)
    mean_fibrilsB = np.mean(interp_fibrilsB, axis=0)

    plt.figure(figsize=(10,5))
    plt.plot(t_grid, mean_fibrilsA, label="Mean fibrils A", color="orange")
    plt.plot(t_grid, mean_fibrilsB, label="Mean fibrils B", color="cornflowerblue")
    plt.xlabel("time")
    plt.ylabel("number of fibrils n")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if distribution == True: 
    
    final_mA = np.array([mA[-1] for mA in all_massesA])
    final_mA_uM = final_mA * 1e6 / (C.N_A * C.V) 
    kde = gaussian_kde(final_mA_uM)
    x_grid = np.linspace(0, final_mA_uM.max()*1.1, 400)
    density = kde(x_grid)

    final_mB = np.array([mB[-1] for mB in all_massesB])
    final_mB_uM = final_mB * 1e6 / (C.N_A * C.V)
    kde2 = gaussian_kde(final_mB_uM)
    x_grid2 = np.linspace(0, final_mB_uM.max()*1.1, 400)
    densityB = kde2(x_grid2)

    plt.figure(figsize=(6,4))
    plt.hist(final_mA_uM, bins=30, density=True,
            color="lightgray", edgecolor="none")
    plt.hist(final_mB_uM, bins=30, density=True,
            color="lightblue", edgecolor="none")
    plt.plot(x_grid, density, color="black", linewidth=2)
    plt.plot(x_grid2, densityB, color="black", linewidth=2)
    plt.xlabel("Fibril mass")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    final_nA = np.array([nA[-1] for nA in all_fibrilsA])
    kdenA = gaussian_kde(final_nA)
    x_gridnA = np.linspace(0, final_nA.max()*1.1, 400)
    densitynA = kdenA(x_gridnA)

    final_nB = np.array([nB[-1] for nB in all_fibrilsB])
    kdenB = gaussian_kde(final_nB)
    x_gridnB = np.linspace(0, final_nB.max()*1.1, 400)
    densitynB = kdenB(x_gridnB)

    plt.figure(figsize=(6,4))
    plt.hist(final_nA, bins=30, density=True,
            color="lightgray", edgecolor="none")
    plt.hist(final_nB, bins=30, density=True,
            color="lightblue", edgecolor="none")
    plt.plot(x_gridnA, densitynA, color="black", linewidth=2)
    plt.plot(x_gridnB, densitynB, color="black", linewidth=2)
    plt.xlabel("Number of Fibrils")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

run = nucleation_events(all_nuclA, all_nuclB)
print(run)