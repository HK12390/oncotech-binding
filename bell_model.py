
#Importing the requisite libraries
import numpy as np
from scipy.optimize import fsolve

def AM1(Nt, params):
    Nb = 0
    A = 0

    N1t = params.N1t
    N2t = params.N2t
    kT = params.kT
    kappa = params.kappa
    tau = params.tau
    L = params.L
    K_L = params.K_L
    gamma = params.gamma
    A1 = params.A1
    A2 = params.A2
    Amax = params.Amax

    def zeta(s):
        return A1 * A2 * Gamma(s) / (kT * K(s))

    def K(s):
        return K_L * np.exp((-0.5 * kappa * (s - L) ** 2) / kT)

    def Gamma(s):
        return (gamma / s) * np.exp(-s / tau)

    S = 0.5 * ( L + (kT / (kappa * tau)) + np.sqrt( (L + (kT / (kappa * tau)))**2 + (4 * kT / kappa) ) )

    def equations(x):
        NB, s = x
        eq1 = (NB / Amax) - ((Nt - NB) / A1) * ((Nt - NB) / A2) * K(s)
        eq2 = (NB / Amax) - (gamma * (s + tau) * np.exp(-s / tau)) / (kappa * tau * s ** 2 * (s - L))
        return [eq1, eq2]

    condition1 = zeta(S) > N1t*N2t
    condition2 = zeta(S) <= (np.minimum(N1t,N2t) - ((Gamma(S)*Amax)/kT) ) * abs(np.maximum(N1t,N2t) - ((Gamma(S)*Amax)/kT))

    if condition1 and not condition2:
        S = 10000 #Some arbitrarily large number for plotting purposes(S -> infinity)
        A = 0
        Nb = 0

    elif condition2 and not condition1:
        A = Amax
        guess = [5.85*10**5, 2.6*10**-6] #Based on the figure in the paper?
        solution = fsolve(equations, np.array(guess))
        Nb = solution[0]
        S = solution[1]

    elif not condition1 and not condition2:
        A = (kT / (2 * Gamma(S))) * (N1t + N2t - np.sqrt((N1t - N2t) ** 2 + 4 * zeta(S)))
        Nb = (1/2) * (N1t + N2t - np.sqrt((N1t - N2t) ** 2 + 4 * zeta(S)))

    #Return dimensionalized values of S,Nb,A
    dimS = ((S/L) - 1)
    dimN = Nb/Nt
    dimA = A/Amax

    return dimS,dimN,dimA




