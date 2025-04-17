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