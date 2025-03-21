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
A = 10**-9 #Contact Region Area (cm^2):   0<= A <= min(A1,A2)

#T-Cell and M@eoA Parameters
N1t =10**4 #Total Receptors in T-cell (~10,000 receptors)
N2t =10**2 + 1  #Total Receptors in M@eoA (~100 receptors)
A1 = 10**-6 #Total Surface Area (cm^2) of T-Cell  (Assuming sphere with r ~ 5000nm)
A2 = 10**-9  #Total Surface Area (cm^2) of M@eoA (Assuming sphere with r ~ 100nm)

#Constants
kT = 4.14*10**-14 #Boltzmann Constant in (dyn*cm^-1) @ T = 300K

#Chemical Potential Cell Bridge Receptor Values 
K = 10**-9 #Binding constant (cm^2) for formation of unstrained cell-cell bridges
k= 0.1 #Force constant (dyn*cm^-1) for stretching of cell-cell bridges 
L = 10**-6 #Optimal Length(cm)

#Glycocalyx Parameters
tau = 10**-6 #Thickness coefficient (cm) of the glycocalys binding constant formation of unstrained cell-cell bridges
gamma = 10**-6 #Compressibility (dyn) coefficient of the glycocalyx

# Create a meshgrid
NB, S = np.meshgrid(Nb, s)

def freeEnergyMinimization(Nb,s):
    n1 = (N1t - Nb)/A1
    n2 = (N2t - Nb)/A2
    
    deltaG = (N1t - Nb)*kT*np.log(n1) + N1t*kT*np.log(N1t/A1) +  (N2t - Nb)*kT*np.log(n2) + N2t*kT*np.log(N2t/A2) + Nb*(kT*(1-np.log(K)+np.log(Nb)) + 1/2*k*(s-L)**2) + (A*gamma/s)*np.exp(-s/tau)                                
    return deltaG/kT

deltaG = freeEnergyMinimization(NB, S)

# Plot the 3D Surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(NB, S, deltaG, cmap='viridis')

ax.set_xlabel('nb (number of bonded receptors)')
ax.set_ylabel('s (separation distance)')
ax.set_zlabel('deltaG')
ax.set_title('3D Surface Plot of deltaG')

plt.show()

#Plot deltaG vs s and deltaG vs nb
