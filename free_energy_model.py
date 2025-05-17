# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:12:24 2025
@author: Hanna
"""
#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt

def free_energy_function(Nb, S, A, params):
    """Computes binding free energy between two cells according to the Bell Model.
    Args:
      params: List of parameters involved in Bell Model.
      Nb: A range of the number of bonded receptors between Cell 1 and Cell 2.
      S: A range of the separation distance between Cell 1 and Cell 2.
      A: The contact region area in cm^-2 of binding between Cell 1 and Cell 2.
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    N1t = params.N1t
    N2t = params.N2t
    A1 = params.A1
    A2 = params.A2
    kT = params.kT
    K_L = params.K_L
    kappa = params.kappa
    L = params.L
    gamma = params.gamma
    tau = params.tau

    deltaG = (N1t - Nb) * kT * np.log((N1t - Nb)/A1) - (N1t * kT * np.log(N1t / A1)) + ((N2t - Nb) * kT * np.log((N2t - Nb)/A2)) - (N2t * kT * np.log(N2t / A2)) + Nb * (kT * (1 - np.log(K_L) + np.log(Nb / A)) + 0.5 * kappa * (S - L) ** 2) + (A * gamma / S) * np.exp(-S / tau)
    return deltaG/kT


def free_energy_function_wrapper(x,params):
    """Computes free_energy_function with x = Nb,S,A to fit with scipy minimize
    Args:
      x: [Nb,S,A]
      params: List of parameters involved in Bell Model.
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    Nb,S,A = x

    n1 = (params.N1t - Nb) / params.A1
    n2 = (params.N2t - Nb) / params.A2

    if n1 <= 0 or n2 <= 0:
        return np.inf  # avoid log(0) or log(negative)

    return free_energy_function(Nb, S, A, params)


def free_energy_function_wrapper_fixedA(x, A, params):
    """Computes free_energy_function with x = Nb,s to fit with scipy minimize
    Args:
      x: [Nb,S]
      A: A fixed constant contact region area.
      params: List of parameters involved in Bell Model.
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    Nb, S = x
    n1 = (params.N1t - Nb) / params.A1
    n2 = (params.N2t - Nb) / params.A2

    if n1 <= 0 or n2 <= 0:
        return np.inf  # avoid log(0) or log(negative)

    return free_energy_function(Nb, S, A, params)

def num_of_bonded_receptors(S, A, params):
    """Computes number of bonded receptors (Nb) of Cell 2 at deltaG = 0 (minimum) for fixed s, fixed A, no glycocalyx effects
    Args:
      S: A fixed separation distance between Cell 1 and Cell 2.
      A: A fixed contact region A between Cell 1 and Cell 2.
      params:
    Returns:
       Nb: A range of the number of bonded receptors between the T-cell and M@eoA
    """
    N2t = params.N2t
    A2 = params.A2
    kT = params.kT
    K_L = params.K_L
    kappa = params.kappa
    L = params.L

    C0 = (K_L * A / A2) * np.exp((-0.5 * (kappa / kT) * (S - L) ** 2) - 1)
    return N2t*(C0/(1+C0))

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

