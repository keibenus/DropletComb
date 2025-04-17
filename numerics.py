# === numerics.py ===
# numerics.py
"""Numerical schemes and helper functions, primarily for FVM."""

import numpy as np
import config

# --- FVM Interpolation Schemes ---

def interpolate_face_value(phi_L, phi_R, u_face, scheme=config.ADVECTION_SCHEME):
    """
    Interpolate value phi at the face using upwind or central difference.
    phi_L: Value in the left cell center
    phi_R: Value in the right cell center
    u_face: Velocity at the face (positive for flow L -> R)
    scheme: 'upwind' or 'central'
    """
    if scheme == 'upwind':
        # If flow is positive (L->R), use left value. If negative (R->L), use right value.
        return phi_L if u_face >= 0.0 else phi_R
    elif scheme == 'central':
        # Simple linear interpolation (average)
        return 0.5 * (phi_L + phi_R)
    else:
        # Default to upwind or raise error if scheme is unknown
        print(f"Warning: Unknown advection scheme '{scheme}', defaulting to 'upwind'.")
        return phi_L if u_face >= 0.0 else phi_R
        # raise ValueError(f"Unknown advection scheme: {scheme}")

# --- FVM Gradient Calculation (at faces, using adjacent cell centers) ---

def gradient_at_face(phi_L, phi_R, r_center_L, r_center_R):
    """Calculate gradient at the face between cells L and R."""
    dr_centers = r_center_R - r_center_L
    # Use a larger epsilon to avoid issues with very close centers
    if abs(dr_centers) < 1e-15:
        # Avoid division by zero if cell centers coincide
        # print("Warning: Cell centers too close for gradient calculation.")
        return 0.0
    else:
        return (phi_R - phi_L) / dr_centers

# --- Property Averaging for Faces ---

def harmonic_mean(val_L, val_R):
    """Calculate harmonic mean, suitable for conductivities/diffusivities."""
    # Add small epsilon to avoid division by zero if val is exactly zero
    eps = 1e-15
    # Handle arrays element-wise
    if isinstance(val_L, np.ndarray) or isinstance(val_R, np.ndarray):
        val_L = np.asarray(val_L)
        val_R = np.asarray(val_R)
        denom = (1.0 / (val_L + eps)) + (1.0 / (val_R + eps))
        # Avoid division by zero in the final step if both inputs were near zero
        result = np.where(denom > eps, 2.0 / denom, 0.0)
    else: # Handle scalars
        denom = (1.0 / (val_L + eps)) + (1.0 / (val_R + eps))
        result = 2.0 / denom if denom > eps else 0.0
    return result

def arithmetic_mean(val_L, val_R):
    """Calculate arithmetic mean."""
    return 0.5 * (val_L + val_R)

# --- Tridiagonal Matrix Solver (Thomas Algorithm) ---
def solve_tridiagonal(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm.
    a: lower diagonal (a[0] is ignored)
    b: main diagonal
    c: upper diagonal (c[-1] is ignored)
    d: right hand side vector
    Returns: solution vector x
    """
    n = len(d)
    if n == 0: return np.array([])
    if n == 1: return d / b if b[0] != 0 else np.zeros(1) # Handle single cell case

    # Modify input arrays in place (or create copies first if needed)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        temp = b[i] - a[i] * c_prime[i-1]
        if abs(temp) < 1e-15: # Avoid division by zero
            print("Error: Zero pivot in Thomas algorithm.")
            return None # Indicate error
        c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / temp

    # Back substitution
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

def calculate_adaptive_dt(
    u_g_faces: np.ndarray,      # Velocity at gas cell faces [m/s] (Ng+1,)
    lambda_g_centers: np.ndarray,# Thermal conductivity at gas cell centers [W/mK] (Ng,)
    rho_g_centers: np.ndarray,   # Density at gas cell centers [kg/m^3] (Ng,)
    cp_g_centers: np.ndarray,    # Specific heat at gas cell centers [J/kgK] (Ng,)
    Dk_g_centers: np.ndarray,   # Diffusion coeffs at gas cell centers [m^2/s] (Nsp, Ng)
    lambda_l_centers: np.ndarray,# Thermal conductivity at liquid cell centers [W/mK] (Nl,)
    rho_l_centers: np.ndarray,   # Density at liquid cell centers [kg/m^3] (Nl,)
    cp_l_centers: np.ndarray,    # Specific heat at liquid cell centers [J/kgK] (Nl,)
    r_g_nodes: np.ndarray,       # Gas face radii [m] (Ng+1,)
    r_l_nodes: np.ndarray,       # Liquid face radii [m] (Nl+1,)
    current_dt: float,           # Previous time step (for limiting increase)
    nsp: int
    ):
    """
    Calculates the maximum stable time step based on CFL conditions.
    """
    if not config.USE_ADAPTIVE_DT:
        return config.DT_INIT # Return fixed small dt if adaptive is off

    # --- Calculate characteristic cell sizes ---
    # Gas phase cell widths (approximate distance between centers)
    dr_g = np.zeros(config.NG)
    if config.NG >= 2:
        dr_g = r_g_nodes[1:] - r_g_nodes[:-1] # Width = face(i+1) - face(i)
    elif config.NG == 1:
        dr_g[0] = r_g_nodes[1] - r_g_nodes[0]
    dr_g = np.maximum(dr_g, 1e-12) # Avoid zero dr

    # Liquid phase cell widths
    dr_l = np.zeros(config.NL)
    if config.NL >= 2:
        dr_l = r_l_nodes[1:] - r_l_nodes[:-1]
    elif config.NL == 1:
        dr_l[0] = r_l_nodes[1] - r_l_nodes[0]
    dr_l = np.maximum(dr_l, 1e-12)

    # Initialize minimum dt allowed by stability to a large value
    dt_stab = config.T_END # Start with a large value

    # --- Advection Limit (Gas) ---
    # Use face velocities and cell widths
    # Need velocity magnitude at cell centers? Approx from faces?
    # Let's use face velocity u_f and cell width dr related to that face
    # dt < CFL * dr / |u|
    if config.NG > 0:
        u_abs_faces = np.abs(u_g_faces[1:-1]) # Internal faces only for dr calculation? Check BCs
        dr_for_u = 0.5 * (dr_g[:-1] + dr_g[1:]) if config.NG >=2 else dr_g # Average dr around internal face
        dt_adv_g = config.CFL_NUMBER * dr_for_u / (u_abs_faces + 1e-9) # Add epsilon for u=0
        if len(dt_adv_g) > 0:
            min_dt_adv_g = np.min(dt_adv_g)
            dt_stab = min(dt_stab, min_dt_adv_g)
        # Also consider boundary faces? u_face[0] might be large
        if config.NG >= 1 and abs(u_g_faces[0]) > 1e-9 :
             dt_adv_0 = config.CFL_NUMBER * dr_g[0] / abs(u_g_faces[0])
             dt_stab = min(dt_stab, dt_adv_0)

    # --- Diffusion Limit (Gas - Thermal) ---
    # dt < CFL * dr^2 / (2 * alpha) where alpha = lambda / (rho * cp)
    if config.NG > 0:
        # Use properties at cell centers and cell width dr_g
        alpha_g = lambda_g_centers / (rho_g_centers * cp_g_centers + 1e-9)
        alpha_g = np.maximum(alpha_g, 1e-12) # Prevent zero alpha
        dt_diff_T_g = 0.5 * config.CFL_NUMBER * dr_g**2 / alpha_g
        min_dt_diff_T_g = np.min(dt_diff_T_g)
        dt_stab = min(dt_stab, min_dt_diff_T_g)

    # --- Diffusion Limit (Gas - Species) ---
    # dt < CFL * dr^2 / (2 * Dk) - find the fastest species diffusion
    if config.NG > 0 and nsp > 0:
        # Use Dk at cell centers and cell width dr_g
        Dk_max_g = np.max(Dk_g_centers, axis=0) # Max Dk at each cell center
        Dk_max_g = np.maximum(Dk_max_g, 1e-12) # Prevent zero Dk
        dt_diff_Y_g = 0.5 * config.CFL_NUMBER * dr_g**2 / Dk_max_g
        #print(dt_diff_Y_g)
        min_dt_diff_Y_g = np.min(dt_diff_Y_g)
        dt_stab = min(dt_stab, min_dt_diff_Y_g)

    # --- Diffusion Limit (Liquid - Thermal) ---
    # dt < CFL * dr^2 / (2 * alpha)
    if config.NL > 0:
        # Use properties at cell centers and cell width dr_l
        alpha_l = lambda_l_centers / (rho_l_centers * cp_l_centers + 1e-9)
        alpha_l = np.maximum(alpha_l, 1e-12) # Prevent zero alpha
        dt_diff_T_l = 0.5 * config.CFL_NUMBER * dr_l**2 / alpha_l
        min_dt_diff_T_l = np.min(dt_diff_T_l)
        dt_stab = min(dt_stab, min_dt_diff_T_l)


    # --- Apply Limits ---
    # Limit the increase from the previous time step
    dt_new = min(dt_stab, current_dt * config.DT_MAX_INCREASE_FACTOR)
    # Apply absolute max and min limits
    dt_new = min(dt_new, config.DT_MAX)
    dt_new = max(dt_new, config.DT_MIN_VALUE) # Ensure dt doesn't become zero

    # Ensure dt doesn't overshoot T_END
    # dt_new = min(dt_new, config.T_END - current_t) # Handled in main loop

    if config.LOG_LEVEL >= 2:
         print(f"DEBUG dt Calc: dt_adv_g={min_dt_adv_g if 'min_dt_adv_g' in locals() else np.inf:.2e} "
               f"dt_diffT_g={min_dt_diff_T_g if 'min_dt_diff_T_g' in locals() else np.inf:.2e} "
               f"dt_diffY_g={min_dt_diff_Y_g if 'min_dt_diff_Y_g' in locals() else np.inf:.2e} "
               f"dt_diffT_l={min_dt_diff_T_l if 'min_dt_diff_T_l' in locals() else np.inf:.2e} -> dt_stab={dt_stab:.2e} -> dt_new={dt_new:.2e}")

    return dt_new




# --- Original FDM Functions (kept for reference or potential reuse if needed) ---
# --- You might want to remove these if FVM is fully adopted ---

def central_diff_1d_fdm(phi, r, dr_left, dr_right):
    """FDM: Second-order central difference for first derivative dphi/dr."""
    Ng = len(phi)
    dphi_dr = np.zeros(Ng)
    if Ng <= 1: return dphi_dr

    # Internal points (i=1 to Ng-2)
    for i in range(1, Ng - 1):
         dr_L = dr_left[i] # r[i] - r[i-1]
         dr_R = dr_right[i] # r[i+1] - r[i]
         dr_T = dr_L + dr_R # r[i+1] - r[i-1]
         if dr_L < 1e-15 or dr_R < 1e-15 or dr_T < 1e-15: continue # Avoid division by zero

         dphi_dr[i] = (phi[i+1] * dr_L**2 +
                       phi[i] * (dr_R**2 - dr_L**2) -
                       phi[i-1] * dr_R**2) / (dr_L * dr_R * dr_T)

    # Boundary points (using second-order one-sided differences)
    if Ng >= 3:
        # At i=0 (using points 0, 1, 2) - Forward difference
        dr01 = dr_right[0]; dr12 = dr_right[1]; dr02 = dr01 + dr12
        if dr01 > 1e-15 and dr02 > 1e-15:
             w0 = -(2*dr01 + dr12) / (dr01 * dr02)
             w1 = (dr01 + dr12) / (dr01 * dr12) # Correction: should be dr02 / (dr01 * dr12)
             w1_corr = dr02 / (dr01 * dr12)
             w2 = -dr01 / (dr12 * dr02)
             # Using standard formula for non-uniform 2nd order forward diff:
             dphi_dr[0] = phi[0]*(-(dr01+dr02)/(dr01*dr02)) + phi[1]*(dr02/(dr01*(dr02-dr01))) - phi[2]*(dr01/(dr02*(dr02-dr01)))

        # At i=Ng-1 (using points Ng-3, Ng-2, Ng-1) - Backward difference
        drN1N2 = dr_left[Ng-1]; drN2N3 = dr_left[Ng-2]; drN1N3 = drN1N2 + drN2N3
        if drN1N2 > 1e-15 and drN1N3 > 1e-15:
            # Using standard formula for non-uniform 2nd order backward diff:
             dphi_dr[Ng-1] = phi[Ng-3]*(drN1N2 / (drN1N3 * (drN1N3 - drN1N2))) - \
                             phi[Ng-2]*(drN1N3 / (drN1N2 * (drN1N3 - drN1N2))) + \
                             phi[Ng-1]*((drN1N2 + drN1N3)/(drN1N2*drN1N3))

    elif Ng == 2: # Use first-order differences
         if dr_right[0] > 1e-15: dphi_dr[0] = (phi[1] - phi[0]) / dr_right[0]
         if dr_left[1] > 1e-15: dphi_dr[1] = (phi[1] - phi[0]) / dr_left[1]

    return dphi_dr

def central_diff_2nd_1d_fdm(phi, r, dr_left, dr_right):
    """FDM: Second-order central difference for second derivative d^2phi/dr^2."""
    Ng = len(phi)
    d2phi_dr2 = np.zeros(Ng)
    if Ng <= 2: return d2phi_dr2 # Cannot compute second derivative

    # Internal points (i=1 to Ng-2)
    for i in range(1, Ng - 1):
         dr_L = dr_left[i]
         dr_R = dr_right[i]
         dr_T = dr_L + dr_R
         if dr_L < 1e-15 or dr_R < 1e-15 or dr_T < 1e-15: continue

         d2phi_dr2[i] = 2 * (phi[i+1] * dr_L -
                            phi[i] * dr_T +
                            phi[i-1] * dr_R) / (dr_L * dr_R * dr_T)

    # Boundary points (Set based on BC or use one-sided approx)
    # Placeholder: Extrapolate from interior (BAD!)
    if Ng >= 3:
        d2phi_dr2[0] = d2phi_dr2[1]
        d2phi_dr2[Ng-1] = d2phi_dr2[Ng-2]
    # Needs proper one-sided 2nd deriv approx if FDM is used at boundary.

    return d2phi_dr2

def central_diff_1d_liquid_fdm(phi, r, dr):
    """FDM: Second-order central diff for liquid (uniform), handles r=0 symmetry."""
    Nl = len(phi)
    dphi_dr = np.zeros(Nl)
    if Nl <= 1: return dphi_dr
    # Internal points
    if Nl >= 3:
        dphi_dr[1:-1] = (phi[2:] - phi[:-2]) / (2 * dr)
    # Boundary r=0 (j=0): Use symmetry dT/dr=0
    dphi_dr[0] = 0.0
    # Boundary r=R (j=Nl-1): Use second-order one-sided backward diff
    if Nl >= 3:
        dphi_dr[Nl - 1] = (3*phi[Nl-1] - 4*phi[Nl-2] + phi[Nl-3]) / (2 * dr)
    elif Nl == 2: # Use first-order backward difference
        if dr > 1e-15: dphi_dr[1] = (phi[1] - phi[0]) / dr
    return dphi_dr