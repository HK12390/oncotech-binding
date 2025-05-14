#Importing the requisite libraries
import numpy as np
import matplotlib.pyplot as plt
import bell_model as bM
import free_energy_model as fM

#Variables
Nt = np.linspace(1, 1*10**6, num=40)

#Empty arrays to store output of function AM1 from BellModel
S_val_AM1 = []
Nb_val_AM1 = []
A_val_AM1 = []

#Section 1: Plotting the analytical solution of the Bell Model using parameters in bell_model.py
for N_value in Nt:
    S_v,Nb_v,A_v = bM.AM1(N_value)
    S_val_AM1.append(S_v)
    Nb_val_AM1.append(Nb_v)
    A_val_AM1.append(A_v)

#Plot the Values
plt.plot(Nt, S_val_AM1, label='S-L/L')
plt.plot(Nt, Nb_val_AM1, label='Nb/N1t')
plt.plot(Nt, A_val_AM1, label='A/Amax')

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
L = 2*10**-6
Amax = 10**-6
Nt = 10**-6

#Step 1: Establish values of s,A and Nb
s_values = np.linspace(10**-6, 10**-5, num=10)
a_values = np.linspace(10**-9,10**-6,num=10)
nb_values = np.linspace(1,10**6,num=10)

#Step 2: Plot deltaG vs Nb/Nt, A/Amax, S/L - 1
deltaG = fM.free_energy_function(s_values, a_values, nb_values, )
print(deltaG)
s_dim_values = (s_values/L) - 1
a_dim_values = a_values/Amax
nb_dim_values = nb_values/Nt

# # Plot 1
# plt.figure()
# plt.plot(s_dim_values, deltaG)
# plt.title('s_dim_values vs deltaG')
# plt.xlabel('s_dim_values')
# plt.ylabel('deltaG')
# plt.grid(True)
#
# # Plot 2
# plt.figure()
# plt.plot(a_dim_values, deltaG)
# plt.title('a_dim_values vs deltaG')
# plt.xlabel('a_dim_values')
# plt.ylabel('deltaG')
# plt.grid(True)
#
# # Plot 3
# plt.figure()
# plt.plot(nb_dim_values, deltaG)
# plt.title('nb_dim_values vs deltaG')
# plt.xlabel('nb_dim_values')
# plt.ylabel('deltaG')
# plt.grid(True)
#
# # Show all plots
# plt.show()

#Step 3: For different values of S/L - 1:
    #Use minimize from scipy to obtain Nb/Nt, A/Amax
    #Plot your results in a similar form to the analytical solution.



