
#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#Table of Parameters
A1 = 2*10**-6
A2 = 2*10**-6
Amax = 10**-6
Nt = np.linspace(1, 1*10**6, num=40)
gamma = 10**-6
tau = 10**-6
K_L = 10**-8
L = 2*10**-6
kappa = 0.1
kT = 4.278*10**-14 #Boltzmann Constant in (ergs) @ T = 310K


#Section 1: The Bell Model Analytically
#Assumption is that N1t = N2t (in other words N1t/N2t = 1)
def AM1(N):
    N1t = N
    N2t = N
    Nb = 0
    A = 0

    S = 0.5 * ( L + (kT / (kappa * tau)) + np.sqrt( (L + (kT / (kappa * tau)))**2 + (4 * kT / kappa) ) )

    def equations(x):
        Nb, S = x
        eq1 = (Nb / Amax) - ((N - Nb) / A1) * ((N - Nb) / A2) * K(S)
        eq2 = (Nb / Amax) - (gamma * (S + tau) * np.exp(-S / tau)) / (kappa * tau * S ** 2 * (S - L))
        return [eq1, eq2]

    condition1 = zeta(S) > N1t*N2t
    condition2 = zeta(S) <= (np.minimum(N1t,N2t) - ((Gamma(S)*Amax)/kT) ) * abs(np.maximum(N1t,N2t) - ((Gamma(S)*Amax)/kT))

    if condition1 and not condition2:
        S = 10000 #Some arbitrarily large number for plotting purposes(S -> infinity)
        A = 0
        Nb = 0

    elif condition2 and not condition1:
        A = Amax
        guess = [7*10**6, 1.3*10**-6] #Based on the figure in the paper?
        solution = fsolve(equations, np.array(guess))
        Nb = solution[0]
        S = solution[1]

    elif not condition1 and not condition2:
        A = (kT / (2 * Gamma(S))) * (N1t + N2t - np.sqrt((N1t - N2t) ** 2 + 4 * zeta(S)))
        Nb = (1/2) * (N1t + N2t - np.sqrt((N1t - N2t) ** 2 + 4 * zeta(S)))

    return S,Nb,A

def zeta(s):
    return A1*A2*Gamma(s)/(kT*K(s))

def K(s):
    return K_L * np.exp( (-0.5*kappa*(s - L)**2)/kT)

def Gamma(s):
    return (gamma/s)*np.exp(-s/tau)

S_values = []
Nb_values = []
A_values = []

for N_value in Nt:
    S_v,Nb_v,A_v = AM1(N_value)
    S_values.append( (S_v/L) - 1)
    Nb_values.append(Nb_v/N_value)
    A_values.append(A_v/Amax)


# Step 4: Plot the Values
plt.plot(Nt, S_values, label='S-L/L')
plt.plot(Nt, Nb_values, label='Nb/N1t')
plt.plot(Nt, A_values, label='A/Amax')

plt.xlabel('N1t = N2t (10^5 molecules/cell)')
plt.ylim(0, 1.2)
tick_positions = np.arange(1e5, 1e6 + 1, 1e5)
tick_labels = [f"{int(tick/1e5)}" for tick in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title('AM1 Model')
plt.legend()
plt.grid(True)
plt.show()


