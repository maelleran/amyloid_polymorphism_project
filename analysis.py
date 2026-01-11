import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import constants as C
from tau_leaping import gillespie_algo

number_trajectories = 5000
t_grid = np.linspace(0, C.t_max, 500)

single = False
many = False
mean = True
dev = True
distribution = True
# ---- single trajectory ----
if single == True:
    times, n_traj, m_traj = gillespie_algo()
    mask = times <= C.t_max
    times = times[mask]
    m = m_traj[mask]
    n = n_traj[mask]
    plt.figure(figsize=(8, 4))
    plt.plot(times/3600, m*1e6/(C.N_A*C.V))
    plt.xlabel("time (h)")
    plt.ylabel("Fibril Mass Concentration (µM)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

all_times = []
all_masses = []
all_fibrils = []

for k in range(number_trajectories):
    t, n_traj, m_traj = gillespie_algo()
    mask = t <= C.t_max
    times = t[mask]
    m = m_traj[mask]
    n = n_traj[mask]
    all_times.append(times)
    all_masses.append(m)
    all_fibrils.append(n)
    if k%200 == 0:
         print(k)

if many == True: # many trajectories plot
 
    plt.figure(figsize=(10, 5))
    for t, m in zip(all_times, all_masses):
        plt.plot(t/3600, m*1e6/(C.N_A*C.V))
    plt.xlabel("time (h)")
    plt.ylabel("Fibril Mass Concentration (µM)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    n_fibrils = False 

    if n_fibrils == True: 

        plt.figure(figsize=(10, 5))

        for t, n in zip(all_times, all_fibrils):
            plt.plot(t/3600, n)

        plt.xlabel("time")
        plt.ylabel("number of fibrils n")
        #plt.title(f"{number_trajectories} Gillespie trajectories — fibril counts")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

if mean == True: # many trajectories and the mean are plotted 

    # Interpolation and mean of the mass
    t_grid = np.linspace(0, C.t_max, 500)
    interp_masses = []

    for t, m in zip(all_times, all_masses):
        interp_m = np.interp(t_grid, t, m)
        interp_masses.append(interp_m)

    interp_masses = np.array(interp_masses)  
    mean_mass = np.mean(interp_masses, axis=0)

    plt.figure(figsize=(10, 5))
    for i in range(number_trajectories):
        plt.plot(t_grid/3600, interp_masses[i]*1e6/(C.N_A*C.V), color="gray", alpha=0.2)
    plt.plot(t_grid/3600, mean_mass*1e6/(C.N_A*C.V), color="black", label="Mean", linewidth=2)
    plt.xlabel("time (h)")
    plt.ylabel("Fibril mass concentration (µM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Interpolation and mean of the number of fibrils 
    interp_fibrils = []
    for t, nA in zip(all_times, all_fibrils):
        interp_f = np.interp(t_grid, t, nA)
        interp_fibrils.append(interp_f)
    interp_fibrils = np.array(interp_fibrils)
    mean_fibrils = np.mean(interp_fibrils, axis=0)

    plt.figure(figsize=(10,5))
    for i in range(number_trajectories):
        plt.plot(t_grid/3600, interp_fibrils[i], color="gray", alpha=0.2)
    plt.plot(t_grid/3600, mean_fibrils, label="Mean fibrils", color="black", linewidth=2)
    plt.xlabel("time (h)")
    plt.ylabel("number of fibrils n")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if dev == True: # mean, median, quartiles (Main 25–75% band) are plotted 
    interp_masses_uM = interp_masses * 1e6 / (C.N_A * C.V)
    mean_mass2 = np.mean(interp_masses_uM, axis=0)
    median_mass = np.median(interp_masses_uM, axis=0)
    q25 = np.percentile(interp_masses_uM, 25, axis=0)
    q75 = np.percentile(interp_masses_uM, 75, axis=0)
    

    plt.figure(figsize=(10, 5))
    plt.fill_between(t_grid, q25, q75, alpha=0.3, label="25–75% quantile")
    plt.plot(t_grid/3600, median_mass, color="black", label="Median", linewidth=2)
    plt.plot(t_grid/3600, mean_mass2, color="blue", alpha= 0.75, label="Mean")
    plt.xlabel("time (h)")
    plt.ylabel("Fibril Mass Concentration (µM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if distribution == True: 
    
    final_m = np.array([m[-1] for m in all_masses])
    final_m_uM = final_m * 1e6 / (C.N_A * C.V) 
    final_n = np.array([n[-1] for n in all_fibrils])
    # Build KDE (smooth density curve)
    kde = gaussian_kde(final_m_uM)
    x_grid = np.linspace(0, final_m_uM.max()*1.1, 400)
    density = kde(x_grid)

    # histogram for m 
    plt.figure(figsize=(6,4))
    plt.hist(final_m_uM, bins=30, density=True,
            color="lightgray", edgecolor="none")
    plt.plot(x_grid, density, color="black", linewidth=2)
    plt.xlabel("Fibril mass concentration (µM)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    kde = gaussian_kde(final_n)
    x_grid = np.linspace(0, final_n.max()*1.1, 400)
    density = kde(x_grid)

    #histogram for n 
    plt.figure(figsize=(6,4))
    plt.hist(final_n, bins=30, density=True,
            color="lightgray", edgecolor="none")
    plt.plot(x_grid, density, color="black", linewidth=2)
    plt.xlabel("Number of Fibrils")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
