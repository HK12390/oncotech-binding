# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:12:24 2025
@author: Hanna
"""
#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Table of Parameters
#T-Cell and M@eoA Parameters
N1t =10**4 #Total Receptors in T-cell (~10,000 receptors)
N2t =10**2 + 1  #Total Receptors in M@eoA (~100 receptors)
A1 = 10**-6 #Total Surface Area (cm^2) of T-Cell  (Assuming sphere with r ~ 5000nm)
A2 = 10**-9  #Total Surface Area (cm^2) of M@eoA (Assuming sphere with r ~ 100nm)

#Constants
kT = 4.278*10**-14 #Boltzmann Constant in (dyn*cm^-1) @ T = 300K

#Chemical Potential Cell Bridge Receptor Values
K_L = 10 ** -8 #Binding constant (cm^2) for formation of unstrained cell-cell bridges
kappa= 0.1 #Force constant (dyn*cm^-1) for stretching of cell-cell bridges
L = 10**-6 #Optimal Length(cm)

#Glycocalyx Parameters
tau = 10**-6 #Thickness coefficient (cm) of the glycocalyx binding constant formation of unstrained cell-cell bridges
gamma = 10**-6 #Compressibility (dyn) coefficient of the glycocalyx

def free_energy_function(Nb, s, A):
    """Computes the free energy of binding between two cells according to the Bell Model.
    Args:
      Nb: A range of the number of bonded receptors between the T-cell and M@eoA
      s: A range of the separation distance between the T-cell and M@eOA as they bind.
      A: The contact region area in cm^-2 of binding between the T-cell and M@eOA
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    deltaG = (N1t - Nb) * kT * np.log((N1t - Nb)/A1) - (N1t * kT * np.log(N1t / A1)) + ((N2t - Nb) * kT * np.log((N2t - Nb)/A2)) - (N2t * kT * np.log(N2t / A2)) + Nb * (kT * (1 - np.log(K_L) + np.log(Nb / A)) + 0.5 * kappa * (s - L) ** 2) + (A * gamma / s) * np.exp(-s / tau)
    return deltaG/kT

def plot_delta_g(NB, S, deltaG, title):
    """Computes the free energy of binding between two cells according to the Bell Model as above, except takes a single parameter x with bundled Nb, S
    to fit with the scipy optimization function
    Args:
    Returns:
      A plot of the free energy required for the receptors to bind vs number of bonded receptors and separation distance.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(NB, S, deltaG, cmap='viridis')
    ax.set_xlabel('nb (number of bonded receptors)')
    ax.set_ylabel('s (separation distance)')
    ax.set_zlabel('deltaG/kT')
    ax.set_title(title)

def free_energy_minimum(x, A):
    """Computes the free energy of binding between two cells according to the Bell Model as above, except takes a single parameter x with bundled Nb, S
    to fit with the scipy optimization function
    Args:
      x: [Nb,s]
      A: A fixed constant contact region area. We are assuming this due to the size differences between the T-cell and the M@eoA
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    Nb = x[0]
    s = x[1]

    # Avoid invalid inputs
    if Nb <= 0 or Nb >= N1t or Nb >= N2t or s <= 0:
        return np.inf  # avoid log(0) or log(negative)

    n1 = (N1t - Nb)/A1
    n2 = (N2t - Nb)/A2

    if n1 <= 0 or n2 <= 0:
        return np.inf  # avoid log(0) or log(negative)

    return free_energy_function(Nb, s, A)

def return_global_minimum(f, x):
    """Computes the global minimum of a function using scipy's minimize
      f: the function to be minimized
      x: a tuple of the variables to be minimized
    Returns:
       min: the global minimum of the function f with respect to variables in x.
    """
    # bounds ensure that a minimum is found between the specified bounds
    bounds = [(1, min(N1t, N2t)), (1e-6, 1e-5)]
    return minimize(f, x, bounds=bounds)


def num_of_bonded_receptors(s, A):
    """Computes the number of bonded receptors (Nb) between two cells according to the Bell Model, when the deltaG = 0 (in other words a minimum) at a fixed separation distance (s)
    Args:
      s: A fixed separation distance between the T-cell and M@eOA as they bind.
      A: A fixed contact region A between the T-cell and M@eOA as they bind.
    Returns:
       Nb: A range of the number of bonded receptors between the T-cell and M@eoA
    """
    #Constant derived from derivative of deltaG expression (freeEnergyMinimization) set to 0, for a fixed s and A.
    C0 = (K_L * A / A2) * np.exp((-0.5 * (kappa / kT) * (s - L) ** 2) - 1)
    return N2t*(C0/(1+C0))


