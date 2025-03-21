# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:12:24 2025
@author: Hanna
"""
#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Table of Parameters
#Variables
Nb = np.linspace(1, 10**2, num=20)  # Range of Number of Bounded Receptors: 0 <= Nb <= min(N1t,N2t)
s = np.linspace(10**-6, 10**-5, num=20)   # Separation Distance (cm) 
A_values = np.linspace(10**-10, 10**-9, num=20)  # A varying over time

#T-Cell and M@eoA Parameters
N1t =10**4 #Total Receptors in T-cell (~10,000 receptors)
N2t =10**2 + 1  #Total Receptors in M@eoA (~100 receptors)
A1 = 10**-6 #Total Surface Area (cm^2) of T-Cell  (Assuming sphere with r ~ 5000nm)
A2 = 10**-9  #Total Surface Area (cm^2) of M@eoA (Assuming sphere with r ~ 100nm)

#Constants
kT = 4.14*10**-14 #Boltzmann Constant in (dyn*cm^-1) @ T = 300K

#Chemical Potential Cell Bridge Receptor Values 
K = 10**-8 #Binding constant (cm^2) for formation of unstrained cell-cell bridges
k= 0.1 #Force constant (dyn*cm^-1) for stretching of cell-cell bridges 
L = 10**-6 #Optimal Length(cm)

#Glycocalyx Parameters
tau = 10**-6 #Thickness coefficient (cm) of the glycocalys binding constant formation of unstrained cell-cell bridges
gamma = 10**-6 #Compressibility (dyn) coefficient of the glycocalyx

# Create a meshgrid
NB, S = np.meshgrid(Nb, s)

def freeEnergyMinimization(Nb,s,A):
    n1 = (N1t - Nb)/A1
    n2 = (N2t - Nb)/A2
    
    deltaG = (N1t - Nb)*kT*np.log(n1) + N1t*kT*np.log(N1t/A1) +  (N2t - Nb)*kT*np.log(n2) + N2t*kT*np.log(N2t/A2) + Nb*(kT*(1-np.log(K)+np.log(Nb)) + 1/2*k*(s-L)**2) + (A*gamma/s)*np.exp(-s/tau)                                
    return deltaG

# Create figure for animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Function to update each frame
def update(frame):
    ax.clear()  # Clear previous frame
    A_var = A_values[frame]  # Get current A value
    
    # Compute ΔG for the current A value
    deltaG_var = freeEnergyMinimization(NB, S, A_var)
    
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
