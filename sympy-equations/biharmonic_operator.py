from sympy import symbols, Function, Derivative, sin, simplify, init_printing
from IPython.display import display

init_printing()

r, phi, theta = symbols('r phi theta')  
f = Function('f')

laplacian = (
    (1/r**2) * Derivative(r**2 * Derivative(f(r, phi, theta), r), r) +
    (1/(r**2 * sin(phi))) * Derivative(sin(phi) * Derivative(f(r, phi, theta), phi), phi) +
    (1/(r**2 * sin(phi)**2)) * Derivative(f(r, phi, theta), theta, theta)
)

lap_phi_only = laplacian.subs({
    Derivative(f(r, phi, theta), r): 0,
    Derivative(f(r, phi, theta), r, r): 0,
    Derivative(f(r, phi, theta), theta): 0,
    Derivative(f(r, phi, theta), theta, theta): 0
}).doit()

bi_laplacian = (
    (1/r**2) * Derivative(r**2 * Derivative(lap_phi_only, r), r) +
    (1/(r**2 * sin(phi))) * Derivative(sin(phi) * Derivative(lap_phi_only, phi), phi) +
    (1/(r**2 * sin(phi)**2)) * Derivative(lap_phi_only, theta, theta)
)

bi_laplacian_phi_only = bi_laplacian.subs({
    Derivative(f(r, phi, theta), r): 0,
    Derivative(f(r, phi, theta), r, r): 0,
    Derivative(f(r, phi, theta), theta): 0,
    Derivative(f(r, phi, theta), theta, theta): 0,
    Derivative(lap_phi_only, r): 0,
    Derivative(lap_phi_only, r, r): 0,
    Derivative(lap_phi_only, theta): 0,
    Derivative(lap_phi_only, theta, theta): 0
}).doit()

simplified_bi_laplacian = simplify(bi_laplacian_phi_only)

display(simplified_bi_laplacian)
