import numpy as np
import free_energy_model as fem

def free_energy_bending_function(Nb_norm, S_norm, theta_norm, params):
    """Computes binding free energy between two cells with bending
    Args:
      params: List of parameters involved in Bell Model.
      Nb_norm: A range of the umber of bonded receptors between Cell 1 and Cell 2.
      S_norm: A range of the separation distance between Cell 1 and Cell 2.
      theta_norm:
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    Nb = Nb_norm * params.N2t
    S = (S_norm + 1) * params.L
    theta = theta_norm * np.pi

    sigma = params.sigma
    R = params.R
    kT = params.kT

    if Nb <= 0 or S <= 0 or theta_norm <= 0 or theta_norm > 1:
        return np.inf

    A_contact = 2*np.pi*R**2*(1 - np.cos(theta))
    deltaG = fem.free_energy_function(Nb, S, A_contact, params)
    deltaG += (4 * np.pi * sigma * (1 - np.cos(theta)))/kT
    return deltaG


def free_energy_bending_wrapper(x, params):
    Nb_norm,S_norm,theta_boundary = x

    n1 = (params.N1t - Nb_norm) / params.A1
    n2 = (params.N2t - Nb_norm) / params.A2

    if n1 <= 0 or n2 <= 0:
        return np.inf

    return free_energy_bending_function(Nb_norm, S_norm, theta_boundary, params)
