# =========================================
#        liquid_phase.py (修正後)
# =========================================
# liquid_phase.py
"""Calculates the RHS for liquid phase energy equation using FVM."""
import numpy as np
import config
from properties import LiquidProperties # For type hinting
from grid import face_areas # Need face areas
from numerics import harmonic_mean, gradient_at_face, solve_tridiagonal # For conductivity averaging and gradient

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

def build_diffusion_matrix_liquid(
    T_l_old: np.ndarray,
    r_l_centers: np.ndarray,
    r_l_nodes: np.ndarray,
    volumes_l: np.ndarray,
    liquid_props: LiquidProperties,
    Nl: int,
    dt: float,
    q_liq_out_surf: float # Heat flux TOWARDS surface FROM liquid [W/m^2]
    ):
    """
    Builds the tridiagonal matrix system for implicit diffusion (Backward Euler).
    Equation: (rho*cp*V/dt) * T_new - (Implicit Diffusion Operator) * T_new = (rho*cp*V/dt) * T_old + Boundary Source
    Returns: a, b, c (diagonals) and d (RHS vector) for Ax=d where x is T_new
    """
    if Nl <= 0: return None, None, None, None

    # Get properties at T_old (or potentially iterate if non-linear)
    rho_l = np.zeros(Nl); cp_l = np.zeros(Nl); lambda_l = np.zeros(Nl)
    valid_props = True
    for j in range(Nl):
        T_val = T_l_old[j].item() if isinstance(T_l_old[j], np.ndarray) else T_l_old[j]
        props = liquid_props.get_properties(T_val)
        rho_l[j] = props.get('density', np.nan)
        cp_l[j] = props.get('specific_heat', np.nan)
        lambda_l[j] = props.get('thermal_conductivity', np.nan)
        if np.isnan(rho_l[j]) or np.isnan(cp_l[j]) or np.isnan(lambda_l[j]): valid_props=False; break
    if not valid_props: print("Error: NaN props in build_diffusion_matrix_liquid"); return None, None, None, None

    A_l_faces = face_areas(r_l_nodes)

    # Matrix diagonals (size Nl)
    a = np.zeros(Nl) # Lower diagonal (a[0] ignored)
    b = np.zeros(Nl) # Main diagonal
    c = np.zeros(Nl) # Upper diagonal (c[Nl-1] ignored)
    d = np.zeros(Nl) # RHS vector

    # --- Calculate Matrix Coefficients ---
    # Coefficient for T_new[j] contribution to its own cell balance (implicit time term)
    diag_coeff = rho_l * cp_l * volumes_l / dt

    # Calculate coefficients related to flux between cells j and j+1 (at face j+1)
    # These coefficients will affect a[j+1], b[j], c[j]
    # Need to calculate for faces j+1 = 1 to Nl (length Nl)
    if Nl >= 1:
        # Interpolate lambda at faces j+1 = 1 to Nl
        lambda_f_p = np.zeros(Nl)
        if Nl == 1: # Single cell, only outer face Nl=1 exists
             # Need lambda at face R (node 1). Average cell 0 prop and surface prop?
             # For simplicity, just use cell 0 prop for face 1 if Nl=1
             lambda_f_p[0] = lambda_l[0]
        else: # Nl >= 2
             # Faces 1 to Nl-1 (internal faces)
             lambda_f_p[:-1] = harmonic_mean(lambda_l[:-1], lambda_l[1:])
             # Face Nl (surface face) - use surface property lambda_l_s
             # We need T_s to get lambda_l_s, which isn't directly available here.
             # Approximate using lambda at last cell center T_l_old[Nl-1]
             lambda_f_p[Nl-1] = lambda_l[Nl-1]

        # Distance between cell centers j and j+1
        dr_c_p = np.zeros(Nl)
        if Nl >= 2:
             dr_c_p[:-1] = r_l_centers[1:] - r_l_centers[:-1]
        # Need dr for face Nl? Use distance between cell Nl-1 and face Nl (R)
        if Nl >= 1:
             dr_c_p[Nl-1] = r_l_nodes[Nl] - r_l_centers[Nl-1] # Approx distance for boundary flux term

        # Avoid division by zero
        dr_c_p = np.maximum(dr_c_p, 1e-15)

        # Calculate the diffusion coefficient term D = lambda * A / dr for faces j+1 = 1 to Nl
        # Note: A_l_faces has length Nl+1 (face 0 to Nl)
        diffusion_coeff_p = lambda_f_p * A_l_faces[1:] / dr_c_p # Length Nl

    # --- Assemble the matrix using coefficients D ---
    # Equation for cell j:
    # diag[j]*Tj_new - ( D_p[j-1]*(Tj_new - Tjm1_new) - D_p[j]*(Tjp1_new - Tj_new) ) = diag[j]*Tj_old + BoundaryFluxTerm
    # Simplified (Backward Euler diffusion):
    # (diag[j] + D_p[j-1] + D_p[j]) * Tj_new - D_p[j-1] * Tjm1_new - D_p[j] * Tjp1_new = diag[j]*Tj_old + BC
    # where D_p[j-1] is diffusion coeff for face j, D_p[j] is for face j+1

    # Lower diagonal a[j] = coefficient of T[j-1] = -D_p[j-1] (for j=1 to Nl-1)
    if Nl >= 2:
        a[1:] = -diffusion_coeff_p[:-1]

    # Upper diagonal c[j] = coefficient of T[j+1] = -D_p[j] (for j=0 to Nl-2)
    if Nl >= 2:
        c[:-1] = -diffusion_coeff_p[:-1]

    # Main diagonal b[j] = diag[j] + D_p[j-1] + D_p[j]
    b[:] = diag_coeff[:] # Start with time derivative term
    if Nl >= 2:
        b[:-1] += diffusion_coeff_p[:-1] # Add term from flux at face j+1 (out)
        b[1:] += diffusion_coeff_p[:-1] # Add term from flux at face j (in)
    elif Nl == 1: # Single cell case
         b[0] += diffusion_coeff_p[0] # Only flux out at face 1

    # --- Right Hand Side (RHS) Vector d ---
    d[:] = diag_coeff[:] * T_l_old[:] # Start with time term

    # Add boundary condition flux terms to RHS
    # Face 0 (r=0): Flux is zero, no modification needed for d[0] or matrix coeffs near j=0

    # Face Nl (r=R): Heat flux INTO liquid q_liq_to_surf * A_l_faces[Nl]
    # This flux enters cell Nl-1. Add it to the RHS of cell Nl-1.
    if Nl >= 1:
        d[Nl-1] = diag_coeff[Nl-1] * T_l_old[Nl-1] + (-q_liq_out_surf) * A_l_faces[Nl]

    # --- Check for NaN ---
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d)):
        print("ERROR: NaN detected in liquid diffusion matrix build.")
        print("a:", a); print("b:", b); print("c:", c); print("d:", d)
        return None, None, None, None

    return a, b, c, d

def calculate_dR_dt(mdot_double_prime, rho_l_s):
     """Calculates dR/dt based on surface mass flux."""
     if rho_l_s > 1e-6:
         dRdt = -mdot_double_prime / rho_l_s
         return dRdt
     else:
         print("Warning: Zero or invalid liquid density at surface in calculate_dR_dt.")
         return 0.0