from dataclasses import dataclass

@dataclass
class BellModelParams:
    N1t: float #Number of Receptors in Cell 1
    N2t: float #Number of Receptors in Cell 2
    A1: float #Surface Area (cm^2) of Cell 1
    A2: float #Surface Area (cm^2) of Cell 2
    Amax: float #Maximum Contact Region Area (cm^2)
    K_L: float #Binding Affinity Constant (cm^2)
    kT: float #Boltzmann Constant in (dyn*cm^-1)
    kappa: float #Force constant (dyn*cm^-1)
    L: float #Optimal Cell Bridge Length(cm)
    gamma: float #Thickness coefficient (cm)
    tau: float #Compressibility (dyn)

my_params = BellModelParams(
    N1t = 10**4, #T-cell
    N2t = 10**2 + 1, #M@eoA
    A1 = 10**-6,
    A2 = 10**-9,
    Amax = 10**-9,
    kT = 4.278*10**-14,
    K_L=10 ** -8,
    kappa = 0.1,
    L = 10**-6,
    gamma = 10**-6,
    tau = 10**-6
)

test_params = BellModelParams(
    N1t = 10**3,
    N2t = 10**6,
    A1 = 2*10**-6,
    A2 = 2*10**-6,
    Amax = 10**-6,
    kT = 4.278*10**-14,
    K_L= 10**-8,
    kappa = 0.1,
    L = 2*10**-6,
    gamma = 10**-6,
    tau = 10**-6
)

