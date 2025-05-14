
import free_energy_model as fem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Variables
Nb = np.linspace(1, 10**2, num=10)  # Range of Number of Bounded Receptors: 0 <= Nb <= min(N1t,N2t)
s = np.linspace(10**-6, 10**-5, num=10)   # Separation Distance (cm)
A_fixed = 10**-9 #Contact Region Area (cm^2):   0<= A <= min(A1,A2)
#Note to edit parameter values refer to the script free_energy_model.py

#Section 1: Plotting the free energy function to visually determine global/local minima of relevance.
# Create a meshgrid
NB, S = np.meshgrid(Nb, s)

deltaG_values = fem.free_energy_function(NB, S, A_fixed)
deltaG_cap =  np.clip(deltaG_values, None, 2000)

# Plot 1: Original deltaG
fem.plot_delta_g(NB, S, deltaG_values, '3D Surface Plot of deltaG')
# Plot 2: Capped deltaG
fem.plot_delta_g(NB, S, deltaG_cap, '3D Surface Plot of deltaG capped')
plt.show()

#Section 2:
# x0 is the initial condition which is used to provide a starting point for the search of a minimum function
x0 = [10, 3*10**-6]

# Wrap the function so it only needs x
wrapped_f = lambda x: fem.free_energy_minimum(x, A_fixed)

# Print the global minimum
globalMinimum = fem.return_global_minimum(wrapped_f, x0)
print("Local minimum at (Nb, s):", globalMinimum.x)
print("Function value at minimum:", globalMinimum.fun)

# #Section 3: Plots of  deltaG vs Nb for varying s, increments of 0.1*10^-5.
for s_value in s:
    deltaG_s = fem.free_energy_function(Nb, s_value, A_fixed)
    plt.figure()  # Create a new figure for each plot
    plt.plot(Nb,deltaG_s)
    plt.title(f"deltaG/kT vs Nb with s = {s_value}")
    plt.xlabel("Nb")
    plt.ylabel("deltaG/kT")
    plt.grid(True)
plt.show()

# #Section 4: Compute the analytical solution from differentiation of freeEnergyFunction with a fixed s, and fixed A.
for s_value in s:
    print(s_value)
    print(fem.num_of_bonded_receptors(s_value, A_fixed))

# #Section 5: Animation of deltaG varying with contact region A.
A_values = np.linspace(10**-10, 10**-9, num=20)  # A varying over time

# Create figure for animation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Function to update each frame for the animation.
def update(frame):
    ax.clear()  # Clear previous frame
    A_var = A_values[frame]  # Get current A value

    # Compute ΔG for the current A value
    deltaG_var = fem.free_energy_function(NB, S, A_var)

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