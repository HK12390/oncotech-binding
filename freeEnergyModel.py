# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:12:24 2025
@author: Hanna
"""
#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.animation as animation

#Table of Parameters
#Variables
Nb = np.linspace(1, 10**2, num=10)  # Range of Number of Bounded Receptors: 0 <= Nb <= min(N1t,N2t)
s = np.linspace(10**-6, 10**-5, num=10)   # Separation Distance (cm) 
A_fixed = 10 ** -9 #Contact Region Area (cm^2):   0<= A <= min(A1,A2)

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

# Create a meshgrid
NB, S = np.meshgrid(Nb, s)



#Section 1: Plotting the free energy function to visually determine global/local minima of relevance.
# Fixed Contact Region (A)
def freeEnergyFunction(Nb,s,A):
    """Computes the free energy of binding between two cells according to the Bell Model.
    Args:
      Nb: A range of the number of bonded receptors between the T-cell and M@eoA
      s: A range of the separation distance between the T-cell and M@eOA as they bind.
    Returns:
      deltaG, the free energy required for the receptors to bind
    """
    deltaG = (N1t - Nb) * kT * np.log((N1t - Nb)/A1) - (N1t * kT * np.log(N1t / A1)) + ((N2t - Nb) * kT * np.log((N2t - Nb)/A2)) - (N2t * kT * np.log(N2t / A2)) + Nb * (kT * (1 - np.log(K_L) + np.log(Nb / A)) + 0.5 * kappa * (s - L) ** 2) + (A * gamma / s) * np.exp(-s / tau)
    return deltaG/kT

deltaG_values = freeEnergyFunction(NB, S, A_fixed)
deltaG_cap =  np.clip(deltaG_values, None, 2000)

def plotDeltaG(NB, S, deltaG, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(NB, S, deltaG, cmap='viridis')
    ax.set_xlabel('nb (number of bonded receptors)')
    ax.set_ylabel('s (separation distance)')
    ax.set_zlabel('deltaG/kT')
    ax.set_title(title)

# Plot 1: Original deltaG
plotDeltaG(NB, S, deltaG_values, '3D Surface Plot of deltaG')
# Plot 2: Capped deltaG
plotDeltaG(NB, S, deltaG_cap, '3D Surface Plot of deltaG capped')
plt.show()



#Section 2: Finding the Global Minimum of the deltaG function with SciPy's minimize
def freeEnergyMinimum(x):
    """Computes the free energy of binding between two cells according to the Bell Model as above, except takes a single parameter x with bundled Nb, S 
    to fit with the scipy optimization function
    Args:
      x: [Nb,s,A]
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

    return freeEnergyFunction(Nb, s, A_fixed)

def returnGlobalMinimum(f,x):
    # bounds ensure that a minimum is found between the specified bounds
    bounds = [(1, min(N1t, N2t)), (1e-6, 1e-5)]
    return minimize(f, x, bounds=bounds)

# x0 is the initial condition which is used to provide a starting point for the search of a minimum function
x0 = [10, 3*10**-6]

# Print the global minimum 
globalMinimum = returnGlobalMinimum(freeEnergyMinimum, x0)
print("Local minimum at (Nb, s):", globalMinimum.x)
print("Function value at minimum:", globalMinimum.fun)



#Section 3: Plots of  deltaG vs Nb for varying s, increments of 0.1*10^-5.
for s_value in s:
    deltaG_s = freeEnergyFunction(Nb,s_value,A_fixed)
    plt.figure()  # Create a new figure for each plot
    plt.plot(Nb,deltaG_s)
    plt.title(f"deltaG/kT vs Nb with s = {s_value}")
    plt.xlabel("Nb")
    plt.ylabel("deltaG/kT")
    plt.grid(True)
plt.show()



#Section 4: Compute the analytical solution from differentiation of freeEnergyFunction with a fixed s, and fixed A.
# In other words, plot deltaG vs Nb
def numOfBondedReceptorsAnalytical(s):
    """Computes the number of bonded receptors (Nb) between two cells according to the Bell Model, when the deltaG = 0 (in other words a minimum) at a fixed separation distance (s)
    Args:
      s: A fixed separation distance between the T-cell and M@eOA as they bind.
    Returns:
       Nb: A range of the number of bonded receptors between the T-cell and M@eoA
    """
    #Constant derived from derivative of deltaG expression (freeEnergyMinimization) set to 0, for a fixed s and A.
    C0 = (K_L * A_fixed / A2) * np.exp((-0.5 * (kappa / kT) * (s - L) ** 2) - 1)
    return N2t*(C0/(1+C0))

# Plot analytical solution and see what you get. 
for s_value in s: 
    print(s_value)
    print(numOfBondedReceptorsAnalytical(s_value))



#Section 5: Animation of deltaG varying with contact region A.
A_values = np.linspace(10**-10, 10**-9, num=20)  # A varying over time

# Create figure for animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Function to update each frame
def update(frame):
    ax.clear()  # Clear previous frame
    A_var = A_values[frame]  # Get current A value

    # Compute ΔG for the current A value
    deltaG_var = freeEnergyFunction(NB, S, A_var)

    # Create 3D surface plot
    ax.plot_surface(NB, S, deltaG_var, cmap='viridis')

    # Labels and title
    ax.set_xlabel('Nb (number of bonded receptors)')
    ax.set_ylabel('s (separation distance)')
    ax.set_zlabel('ΔG (Free Energy)')
    ax.set_title(f'3D Plot of ΔG (A = {A_var:.1e})')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(A_values), interval=300, blit=False)
# Save animation as a GIF
ani.save("deltaG_animation.gif", writer='pillow', fps=5)

