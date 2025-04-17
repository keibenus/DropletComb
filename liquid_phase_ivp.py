# =========================================
#        liquid_phase.py (修正後)
# =========================================
# liquid_phase.py
"""Calculates the RHS for liquid phase energy equation using FVM."""
import numpy as np
import config
from properties import LiquidProperties # For type hinting
from grid import face_areas # Need face areas
from numerics import harmonic_mean, gradient_at_face # For conductivity averaging and gradient

def calculate_dTl_dt_fvm(
    T_l_centers: np.ndarray, # Temperature at cell centers [K]
    r_l_centers: np.ndarray, # Cell center radii [m]
    r_l_nodes: np.ndarray, # Cell face radii [m]
    volumes_l: np.ndarray, # Cell volumes [m^3]
    liquid_props: LiquidProperties,
    Nl: int,
    q_liq_to_surf: float # Heat flux FROM liquid TO the surface face (r=R, node Nl) [W/m^2]
    ):
    """
    Calculates dTl/dt using Finite Volume Method for the liquid phase energy equation.
    """
    # ... (Property calculation remains the same) ...
    dTl_dt = np.zeros(Nl)
    if Nl <= 0: return dTl_dt

    rho_l = np.zeros(Nl); cp_l = np.zeros(Nl); lambda_l = np.zeros(Nl)
    valid_props = True
    for j in range(Nl):
        T_val = T_l_centers[j].item() if isinstance(T_l_centers[j], np.ndarray) else T_l_centers[j]
        props = liquid_props.get_properties(T_val)
        rho_l[j] = props.get('density', np.nan)
        cp_l[j] = props.get('specific_heat', np.nan)
        lambda_l[j] = props.get('thermal_conductivity', np.nan)
        if np.isnan(rho_l[j]) or np.isnan(cp_l[j]) or np.isnan(lambda_l[j]):
            print(f"Error: NaN liquid properties in cell {j} (T={T_val:.1f}K).")
            valid_props = False; break
    if not valid_props: return np.zeros(Nl)
    #print(T_val,rho_l,cp_l,lambda_l)
    A_l_faces = face_areas(r_l_nodes) # Area at each face [m^2]

    # --- Flux Calculation (Diffusion only) ---
    # Heat flux rate Q = q * A [W] across each face.
    # Positive Q means heat flowing TOWARDS the center (in -r direction).
    Q_l_faces_W = np.zeros(Nl + 1)

    if Nl >= 2:
        lambda_l_face_internal = harmonic_mean(lambda_l[:-1], lambda_l[1:])
        # Calculate gradient between cell centers to find flux at face between them
        grad_T_face = np.array([gradient_at_face(T_l_centers[j-1], T_l_centers[j], r_l_centers[j-1], r_l_centers[j]) for j in range(1, Nl)])
        # Flux rate towards center at internal faces (nodes 1 to Nl-1)
        Q_l_faces_W[1:Nl] = -lambda_l_face_internal * grad_T_face * A_l_faces[1:Nl]

    Q_l_faces_W[0] = 0.0 # Symmetry

    # Boundary face at r=R (node Nl)
    # q_liq_to_surf is defined positive TOWARDS surface.
    # We need the flux rate LEAVING the liquid, which is Q = q_liq_to_surf * A
    Q_l_faces_W[Nl] = q_liq_to_surf * A_l_faces[Nl]

    # --- Calculate RHS for each cell ---
    # rho*cp*Vol * dT/dt = Rate_Energy_In - Rate_Energy_Out
    # Rate_Energy_In comes from face j+1 (outer face, flux defined positive towards center)
    # Rate_Energy_Out goes to face j (inner face, flux defined positive towards center)
    # Correct energy balance: dE/dt = Q[j] - Q[j+1] (if Q is positive towards center)
    # Or dE/dt = Flux_at_inner_face - Flux_at_outer_face (if flux is positive outwards)
    # Let's redefine Q_l_faces_W to be positive outwards for consistency with gas phase?
    # Let Q_out[j] be the flux rate leaving cell j through face j+1 (outward)
    # Let Q_in[j] be the flux rate entering cell j through face j (outward from j-1)
    # dE/dt = Q_in[j] - Q_out[j]
    Q_outward = np.zeros(Nl + 1) # Flux rate positive in +r direction
    if Nl >= 2:
         lambda_l_face_internal = harmonic_mean(lambda_l[:-1], lambda_l[1:])
         grad_T_face = np.array([gradient_at_face(T_l_centers[j-1], T_l_centers[j], r_l_centers[j-1], r_l_centers[j]) for j in range(1, Nl)])
         Q_outward[1:Nl] = -lambda_l_face_internal * grad_T_face * A_l_faces[1:Nl] # Check sign again

    Q_outward[0] = 0.0 # No flux out at center
    # Flux leaving liquid at r=R is q_liq_to_surf * A (positive if Tl > Ts)
    Q_outward[Nl] = q_liq_to_surf * A_l_faces[Nl] # Flux leaving liquid phase

    for j in range(Nl):
        flux_in_W = Q_outward[j]   # Flux entering cell j from face j
        flux_out_W = Q_outward[j+1] # Flux leaving cell j from face j+1

        denominator = rho_l[j] * cp_l[j] * volumes_l[j]
        if denominator > 1e-12:
            dTl_dt[j] = (flux_in_W - flux_out_W) / denominator
        else:
            dTl_dt[j] = 0.0

    return dTl_dt


def calculate_dR_dt(mdot_double_prime, rho_l_s):
     """Calculates dR/dt based on surface mass flux."""
     if rho_l_s > 1e-6:
         dRdt = -mdot_double_prime / rho_l_s
         return dRdt
     else:
         print("Warning: Zero or invalid liquid density at surface in calculate_dR_dt.")
         return 0.0