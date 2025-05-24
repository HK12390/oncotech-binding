#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
import bell_model as bem
import free_energy_model as fem
from config import test_params, BellModelParams
from scipy.optimize import minimize

#Variables
Nt = np.linspace(1, 1*10**6, num=50)

#Empty arrays to store output of function AM1 from BellModel
S_AM1 = []
Nb_AM1 = []
A_AM1 = []

#Section 1: Plotting the analytical solution of the Bell Model using parameters in bell_model.py
for N in Nt:
    dynamic_params = BellModelParams(
        N1t=N,
        N2t=N,
        A1=test_params.A1,
        A2=test_params.A2,
        Amax=test_params.Amax,
        kT=test_params.kT,
        K_L=test_params.K_L,
        kappa=test_params.kappa,
        L=test_params.L,
        gamma=test_params.gamma,
        tau=test_params.tau
    )
    S,Nb,A = bem.AM1(N, params = dynamic_params)
    S_AM1.append(S)
    Nb_AM1.append(Nb)
    A_AM1.append(A)

#Plot the Values
plt.plot(Nt, Nb_AM1, label='Nb/N1t')
plt.plot(Nt, S_AM1, label='S-L/L')
plt.plot(Nt, A_AM1, label='A/Amax')

plt.xlabel('N1t = N2t (10^5 molecules/cell)')
plt.ylim(0, 1.2)
tick_positions = np.arange(1e5, 1e6 + 1, 1e5)
tick_labels = [f"{int(tick/1e5)}" for tick in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title('AM1 Model')
plt.legend()
plt.grid(True)
plt.show()

# #Section 2: Plotting the numerical solution of the Bell Model using parameters in bell_model.py
#Constants for Dimensionality

#Step 1: Empty arrays
S_num = []
A_num = []
Nb_num = []

for N in Nt:
    dynamic_params = BellModelParams(
        N1t=N,
        N2t=N,
        A1=test_params.A1,
        A2=test_params.A2,
        Amax=test_params.Amax,
        kT=test_params.kT,
        K_L=test_params.K_L,
        kappa=test_params.kappa,
        L=test_params.L,
        gamma=test_params.gamma,
        tau=test_params.tau
    )
    # Define bounds based on dynamic parameters
    Nb_min = 1e-3
    Nb_max = N - 1e-3
    S_bounds = (1e-10, 1e-5)
    A_bounds = (8e-13, dynamic_params.Amax)

    bounds = [ (Nb_min, Nb_max), S_bounds, A_bounds ]

    # Safe initial guess within bounds
    x0 = np.array([
        0.5 * (Nb_min + Nb_max),   # Midpoint for Nb
        0.5 * (S_bounds[0] + S_bounds[1]),  # Midpoint for S
        0.5 * (A_bounds[0] + A_bounds[1])   # Midpoint for A
    ])

    minimum = minimize(
        fem.free_energy_function_wrapper,
        x0,
        args=(dynamic_params,),
        method='nelder-mead',
        bounds= bounds)
    Nb, S, A = minimum.x

    Nb_num.append(Nb / N)
    S_num.append((S / dynamic_params.L) - 1)
    A_num.append(A / dynamic_params.Amax)

plt.plot(Nt, Nb_num, label='Nb/N1t')
plt.plot(Nt, S_num, label='S-L/L')
plt.plot(Nt, A_num, label='A/Amax')

plt.xlabel('N1t = N2t (10^5 molecules/cell)')
plt.ylim(0, 1.2)
tick_positions = np.arange(1e5, 1e6 + 1, 1e5)
tick_labels = [f"{int(tick/1e5)}" for tick in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title('Numerical Model')
plt.legend()
plt.grid(True)
plt.show()










