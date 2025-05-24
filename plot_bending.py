import bending_model as bm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from config import bending_params
from dataclasses import replace

#Variables
bending_modulus_values = np.linspace(10**-13,10**-10,num=40)

#Step 1: Empty arrays
S_num = []
Nb_num = []
theta_num = []

for bending_modulus in bending_modulus_values:
    dynamic_params = replace(bending_params, sigma=bending_modulus)

    Nb_bounds = (0.01,1.0)
    S_bounds = (-0.25, 1.0)
    theta_bounds = (0.0001,1.0)

    bounds = [Nb_bounds, S_bounds, theta_bounds]

    x0 = np.array([
        0.5 * (Nb_bounds[0] + Nb_bounds[1]),   # Midpoint for Nb
        0.5 * (S_bounds[0] + S_bounds[1]),  # Midpoint for S
        0.5 * (theta_bounds[0] + theta_bounds[1])   # Midpoint for A
    ])

    minimum = minimize(
        bm.free_energy_bending_wrapper,
        x0,
        args=(dynamic_params,),
        method='nelder-mead',
        bounds= bounds)
    Nb, S, theta = minimum.x

    Nb_num.append(Nb)
    S_num.append(S)
    theta_num.append(theta)


plt.plot(bending_modulus_values, Nb_num, label='Nb/N2t')
plt.plot(bending_modulus_values, S_num, label='(S-L)/L')
plt.plot(bending_modulus_values, theta_num, label='theta/pi')

plt.xlabel('bending modulus (10^-13 ergs)')
plt.ylim(0, 1.2)
plt.xlabel('Bending modulus (erg)')

plt.title('Bending Model')
plt.legend()
plt.grid(True)
plt.show()