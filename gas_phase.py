# =========================================
#          gas_phase.py (修正後)
# =========================================
# gas_phase.py
"""Calculates the RHS for gas phase equations using FVM."""
import numpy as np
import config
import properties
from properties import GasProperties # For type hinting
from grid import face_areas # Need face areas
from numerics import interpolate_face_value, gradient_at_face, harmonic_mean, arithmetic_mean
import reactions # Import reactions module

def calculate_gas_rhs_fvm(
    T_g_centers: np.ndarray, # Temperature at cell centers [K]
    Y_g_centers: np.ndarray, # Mass fractions at cell centers [nsp, Ng]
    rho_g_centers: np.ndarray, # Density at cell centers [kg/m^3]
    P: float, # Pressure (Pa)
    R_current: float, # Current droplet radius (m)
    r_g_centers: np.ndarray, # Cell center radii [m]
    r_g_nodes: np.ndarray, # Cell face radii [m]
    volumes_g: np.ndarray, # Cell volumes [m^3]
    gas_props: GasProperties,
    Nl: int, Ng: int, nsp: int,
    mdot_double_prime: float, # Mass flux leaving surface (r=R) [kg/m^2/s] > 0 for evap
    q_gas_to_surf: float, # Heat flux FROM gas TO surface face (r=R) [W/m^2] > 0 if Tg>Ts
    v_g_surf: float, # Gas velocity normal TO surface (r=R) [m/s] > 0 for evap
    Y_equilibrium: np.ndarray # Equilibrium mass fractions at surface [-]
    ):
    """
    Calculates dTg/dt and dYg/dt using Finite Volume Method.
    Handles boundary fluxes explicitly. Corrects NameError.
    """
    dTg_dt = np.zeros(Ng)
    dYg_dt = np.zeros((nsp, Ng))
    if Ng <= 0: return dTg_dt, dYg_dt

    fuel_idx = gas_props.fuel_idx

    # --- 0. Calculate properties at cell centers ---
    cp_g = np.zeros(Ng); lambda_g = np.zeros(Ng); h_k_g = np.zeros((nsp, Ng))
    Dk_g = np.zeros((nsp, Ng))
    valid_props = True
    for i in range(Ng):
         if not gas_props.set_state(T_g_centers[i], P, Y_g_centers[:, i]):
             print(f"Warning: Failed set state gas cell {i} (T={T_g_centers[i]:.1f}). Using NaNs.")
             cp_g[i]=np.nan; lambda_g[i]=np.nan; h_k_g[:, i]=np.nan; Dk_g[:, i]=np.nan
             valid_props = False; continue
         cp_g[i] = gas_props.gas.cp_mass
         lambda_g[i] = gas_props.gas.thermal_conductivity
         h_k_g[:, i] = gas_props.get_partial_enthalpies_mass(T_g_centers[i], P, Y_g_centers[:, i])
         Dk_g[:, i] = gas_props.get_diffusion_coeffs(T_g_centers[i], P, Y_g_centers[:, i], option=config.DIFFUSION_OPTION)
         if np.isnan(cp_g[i]) or np.isnan(lambda_g[i]) or np.isnan(h_k_g[:,i]).any() or np.isnan(Dk_g[:,i]).any():
             print(f"Warning: NaN props calculated gas cell {i} (T={T_g_centers[i]:.1f}). Filling.")
             cp_g[i]=np.nan_to_num(cp_g[i], nan=1000.0); lambda_g[i]=np.nan_to_num(lambda_g[i], nan=0.1)
             h_k_g[:, i]=np.nan_to_num(h_k_g[:, i], nan=0.0); Dk_g[:, i]=np.nan_to_num(Dk_g[:, i], nan=1e-6)

    # --- 1. Calculate Velocity Field 'u_g' at faces ---
    A_g_faces = face_areas(r_g_nodes)
    u_g_faces = np.zeros(Ng + 1)
    mass_flux_faces = np.zeros(Ng + 1) # Mass flux rate [kg/s] = rho*u*A, positive outwards

    # Mass flux rate at the surface face (face 0)
    mass_flux_faces[0] = mdot_double_prime * A_g_faces[0]
    # --- Density at face 0 (r=R) --- CORRECTION ---
    # Use surface properties (Ts, Yeq) and cell 0 properties
    T_s = T_g_centers[Nl-1] if config.NL> 0 else T_g_centers[0] # Estimate Ts from last liquid or first gas cell
    rho_g_s = gas_props.get_density(T_s, P, Y_equilibrium) # Density at surface conditions
    rho_face_0 = arithmetic_mean(rho_g_s, rho_g_centers[0]) # Average with first cell center
    # --- End Correction ---
    if A_g_faces[0] > 1e-15 and rho_face_0 > 1e-6:
        u_g_faces[0] = mass_flux_faces[0] / (rho_face_0 * A_g_faces[0])
    else: u_g_faces[0] = 0.0

    # Integrate outwards assuming quasi-steady mass flow rate Mdot = constant
    Mdot = mass_flux_faces[0]
    for i in range(1, Ng):
         mass_flux_faces[i] = Mdot # kg/s outwards through face i
         rho_face_i = arithmetic_mean(rho_g_centers[i-1], rho_g_centers[i])
         if A_g_faces[i] > 1e-15 and rho_face_i > 1e-6:
             u_g_faces[i] = mass_flux_faces[i] / (rho_face_i * A_g_faces[i])
         else:
             u_g_faces[i] = 0.0

    mass_flux_faces[Ng] = 0.0 # Outer boundary mass flux is zero
    u_g_faces[Ng] = 0.0

    # --- 2. Calculate Reaction Rates ---
    if config.REACTION_TYPE == 'detailed':
        wdot = reactions.calculate_cantera_rates(gas_props.gas, T_g_centers, P, Y_g_centers)
    elif config.REACTION_TYPE == 'overall':
        wdot = reactions.calculate_overall_rates(rho_g_centers, T_g_centers, Y_g_centers, gas_props.gas, P)
    else: wdot = np.zeros((nsp, Ng))

    # --- 3. Calculate Total Fluxes at Cell Faces ---
    # Total Flux = Advection + Diffusion (Positive = Outwards)
    HeatFlux_faces = np.zeros(Ng + 1) # [W]
    SpeciesFlux_faces = np.zeros((nsp, Ng + 1)) # [kg/s]

    # Iterate through internal faces (i = 1 to Ng-1)
    for i in range(1, Ng):
        # L = cell i-1, R = cell i
        T_L = T_g_centers[i-1]; T_R = T_g_centers[i]
        Y_L = Y_g_centers[:, i-1]; Y_R = Y_g_centers[:, i]
        rho_L = rho_g_centers[i-1]; rho_R = rho_g_centers[i]
        h_k_L = h_k_g[:, i-1]; h_k_R = h_k_g[:, i]

        lambda_f = harmonic_mean(lambda_g[i-1], lambda_g[i])
        Dk_f = harmonic_mean(Dk_g[:, i-1], Dk_g[:, i])
        rho_f = arithmetic_mean(rho_L, rho_R)

        u_f = u_g_faces[i]
        mass_flux_rate = mass_flux_faces[i] # kg/s

        # Advected values at face
        Yk_adv = np.array([interpolate_face_value(Y_L[k], Y_R[k], u_f) for k in range(nsp)])
        hk_adv = np.array([interpolate_face_value(h_k_L[k], h_k_R[k], u_f) for k in range(nsp)])
        H_adv_flux = mass_flux_rate * np.sum(Yk_adv * hk_adv)
        Yk_adv_flux = mass_flux_rate * Yk_adv

        # Diffusion gradients and fluxes
        grad_T = gradient_at_face(T_L, T_R, r_g_centers[i-1], r_g_centers[i])
        grad_Yk = np.array([gradient_at_face(Y_L[k], Y_R[k], r_g_centers[i-1], r_g_centers[i]) for k in range(nsp)])
        # Diffusive species flux rate [kg/s] (Fick's law)
        Yk_diff_flux = -rho_f * Dk_f * grad_Yk * A_g_faces[i]
        # Heat flux due to conduction + enthalpy diffusion [W]
        H_cond_flux = -lambda_f * grad_T * A_g_faces[i]
        H_sp_diff_flux = np.sum(hk_adv * Yk_diff_flux) # Enthalpy carried by species diffusion
        H_diff_flux_corr = H_cond_flux + H_sp_diff_flux

        HeatFlux_faces[i] = H_adv_flux + H_diff_flux_corr
        SpeciesFlux_faces[:, i] = Yk_adv_flux + Yk_diff_flux

    # --- 4. Boundary Fluxes ---
    # --- Face 0 (at r = R) --- Flux OUT of gas cell 0
    # Heat Flux: Heat arriving from gas MINUS heat used for species diffusion away
    # q_gas_to_surf was heat flux TO surface from gas = - (conduction + species_diffusion_h)
    # So, H_diff_flux_corr[0] = -q_gas_to_surf * A_g_faces[0] ? No.
    # Let's define boundary fluxes directly based on interface calc.
    # Total energy flux leaving gas cell 0 = enthalpy advected away + heat conducted away
    # HeatFlux_faces[0] should be net energy leaving cell 0 through face 0.
    # Energy balance at face 0: Flux_Gas_to_Surf = q_gas_to_surf
    # This energy is transferred via conduction and species enthalpy diffusion.
    # We also have enthalpy advected away by mdot''.
    H_adv_face0 = mass_flux_faces[0] * np.sum(Y_equilibrium * h_k_g[:,0]) # Enthalpy advected at T_g[0] with Y_eq? Use Ts?
    # Need enthalpy at Ts, Yeq
    gas_props.set_state(T_s, P, Y_equilibrium)
    hk_s = gas_props.get_partial_enthalpies_mass(T_s, P, Y_equilibrium)
    H_adv_face0_corr = mass_flux_faces[0] * np.sum(Y_equilibrium * hk_s) # Enthalpy leaving with mdot'' at T_s

    # Heat flux by conduction TOWARDS surface is q_gas_to_surf - sum(hk * Jk_diff)
    # Total Heat Flux = Advection + Conduction + Species Enthalpy Diffusion
    # This needs careful definition. Let's use the simpler interface values.
    # Net energy INTO cell 0 from surface = q_gas_to_surf * A[0]
    # The FVM RHS needs Flux_Out[0] - Flux_In[0] ... let's stick to Flux[i] - Flux[i+1]
    # So, HeatFlux_faces[0] = Energy flux LEAVING cell 0 towards interface.
    HeatFlux_faces[0] = -q_gas_to_surf * A_g_faces[0] # Heat conducted+diffused TOWARDS surface

    # Species Flux: Total net flux OUT of cell 0 is mdot'' * Yk_leaving
    SpeciesFlux_faces[fuel_idx, 0] = mdot_double_prime * A_g_faces[0] # Fuel leavingcalculate_gas_rhs_fvm
    for k in range(nsp):
        if k != fuel_idx: SpeciesFlux_faces[k, 0] = 0.0 # Others don't leave

    # --- Face Ng (at r = rmax) ---
    HeatFlux_faces[Ng] = 0.0
    SpeciesFlux_faces[:, Ng] = 0.0

    # --- 5. Calculate RHS for each cell ---
    for i in range(Ng):
        if rho_g_centers[i] < 1e-12 or volumes_g[i] < 1e-30: continue

        # Energy Equation
        net_energy_flux_W = HeatFlux_faces[i] - HeatFlux_faces[i+1] # In - Out
        source_energy_W_m3 = -np.sum(h_k_g[:, i] * wdot[:, i])
        source_energy_W = source_energy_W_m3 * volumes_g[i]
        denom_T = rho_g_centers[i] * cp_g[i] * volumes_g[i]
        if denom_T > 1e-15:
            dTg_dt[i] = (net_energy_flux_W + source_energy_W) / denom_T
        else: dTg_dt[i] = 0.0

        # Species Equations
        net_species_flux_kgs = SpeciesFlux_faces[:, i] - SpeciesFlux_faces[:, i+1] # In - Out
        source_species_kgs_m3 = wdot[:, i]
        source_species_kgs = source_species_kgs_m3 * volumes_g[i]
        denom_Y = rho_g_centers[i] * volumes_g[i]
        if denom_Y > 1e-20:
             dYg_dt[:, i] = (net_species_flux_kgs + source_species_kgs) / denom_Y
        else: dYg_dt[:, i] = 0.0

    # --- Sanity check & Correction ---
    dYg_dt = np.nan_to_num(dYg_dt, nan=0.0)
    dTg_dt = np.nan_to_num(dTg_dt, nan=0.0)

    sum_dYdt = np.sum(dYg_dt, axis=0)
    if np.any(np.abs(sum_dYdt) > 1e-3):
        if config.LOG_LEVEL >= 1: print(f"Warning: Sum dYg/dt non-zero (max abs = {np.max(np.abs(sum_dYdt)):.2e}). Adjusting N2...")
        if gas_props.n2_idx >= 0:
             dYg_dt[gas_props.n2_idx, :] -= sum_dYdt
        # else: print("Warning: N2 index not found for conservation correction.")

    return dTg_dt, dYg_dt

# --- Function for Explicit Advection Term Calculation ---
def calculate_gas_advection_rhs(
    T_g: np.ndarray, Y_g: np.ndarray, rho_g: np.ndarray, cp_g: np.ndarray, h_k_g: np.ndarray,
    u_g_faces: np.ndarray, # Velocities at faces
    r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, nsp: int
    ):
    """Calculates the RHS contribution from advection only."""
    dT_dt_adv = np.zeros(Ng)
    dY_dt_adv = np.zeros((nsp, Ng))
    if Ng <= 0: return dT_dt_adv, dY_dt_adv

    A_g_faces = face_areas(r_g_nodes)

    # Allocate flux arrays
    AdvFlux_H = np.zeros(Ng + 1) # Advected enthalpy flux rate [W]
    AdvFlux_Yk = np.zeros((nsp, Ng + 1)) # Advected species mass flux rate [kg/s]

    # Calculate fluxes through internal faces (i=1 to Ng-1)
    for i in range(1, Ng):
        u_f = u_g_faces[i]
        mass_flux_rate = rho_g[i-1] * u_f * A_g_faces[i] if u_f >= 0 else rho_g[i] * u_f * A_g_faces[i] # Approx rho@face*u*A
        # Use simpler arithmetic mean for density at face
        rho_f = arithmetic_mean(rho_g[i-1], rho_g[i])
        mass_flux_rate = rho_f * u_f * A_g_faces[i] # Corrected mass flux rate

        Yk_adv = np.array([interpolate_face_value(Y_g[:, i-1][k], Y_g[:, i][k], u_f) for k in range(nsp)])
        hk_adv = np.array([interpolate_face_value(h_k_g[:, i-1][k], h_k_g[:, i][k], u_f) for k in range(nsp)])

        AdvFlux_H[i] = mass_flux_rate * np.sum(Yk_adv * hk_adv)
        AdvFlux_Yk[:, i] = mass_flux_rate * Yk_adv

    # Boundary Face 0 (r=R) - Advection flux defined by mdot'' and Y_equilibrium
    # Need enthalpy at surface state
    # This requires Ts and Yeq - assume they are handled outside or approximated
    # For now, assume AdvFlux_H[0] and AdvFlux_Yk[:,0] are handled by the main flux calculation logic

    # Boundary Face Ng (r=rmax) - Zero advection flux
    AdvFlux_H[Ng] = 0.0
    AdvFlux_Yk[:, Ng] = 0.0

    # Calculate RHS for each cell
    for i in range(Ng):
        if rho_g[i] < 1e-9 or volumes_g[i] < 1e-25: continue
        # Net Advective Flux = Flux_in[i] - Flux_out[i+1]
        net_adv_H = AdvFlux_H[i] - AdvFlux_H[i+1]
        net_adv_Yk = AdvFlux_Yk[:, i] - AdvFlux_Yk[:, i+1]

        if cp_g[i] > 1e-6: dT_dt_adv[i] = net_adv_H / (rho_g[i] * cp_g[i] * volumes_g[i])
        dY_dt_adv[:, i] = net_adv_Yk / (rho_g[i] * volumes_g[i])

    return dT_dt_adv, dY_dt_adv

# --- Functions for Implicit Diffusion Matrix ---
def build_diffusion_matrix_gas_T(
    T_g_star: np.ndarray, # Temp after advection step (used for RHS)
    rho_g: np.ndarray, cp_g: np.ndarray, lambda_g: np.ndarray, # Props at cell centers (from T_old)
    r_g_centers: np.ndarray, r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, dt: float, # nsp は不要
    q_gas_out_surf: float, # Heat flux outword from gas is positive at r=R boundary [W/m^2]
    A_g_faces: np.ndarray # Precomputed face areas
    ):
    """Builds tridiagonal matrix for implicit heat diffusion (Backward Euler)."""
    if Ng <= 0: return None, None, None, None
    a, b, c = np.zeros(Ng), np.zeros(Ng), np.zeros(Ng)
    d = np.zeros(Ng) # RHS vector

    # Check for NaN/Inf in inputs
    if np.any(np.isnan(rho_g)) or np.any(np.isnan(cp_g)) or np.any(np.isnan(lambda_g)) or np.any(np.isnan(T_g_star)):
        print("ERROR: NaN input detected in build_diffusion_matrix_gas_T")
        return None, None, None, None

    # Coefficient for T_new[i] contribution to its own cell balance
    diag_coeff = rho_g * cp_g * volumes_g / dt

    # Calculate diffusion coefficient term D = lambda * A / dr for faces i+1 = 1 to Ng
    if Ng >= 1:
        diffusion_coeff_p = np.zeros(Ng) # Coeff related to face i+1 (length Ng)

        # Internal faces (i+1 = 1 to Ng-1) -> affects cells i=0 to Ng-2
        if Ng >= 2:
            lambda_f_internal = harmonic_mean(lambda_g[:-1], lambda_g[1:]) # Faces 1 to Ng-1
            dr_c_internal = r_g_centers[1:] - r_g_centers[:-1] # Distances for faces 1 to Ng-1
            dr_c_internal = np.maximum(dr_c_internal, 1e-15) # Avoid zero division
            diffusion_coeff_p[:-1] = lambda_f_internal * A_g_faces[1:Ng] / dr_c_internal

        # Outer face (i+1 = Ng) - Flux is zero here (Neumann boundary)
        diffusion_coeff_p[Ng-1] = 0.0 # No diffusive flux out at rmax

        diffusion_coeff_p = np.nan_to_num(diffusion_coeff_p) # Clean up potential NaNs


    # --- Assemble the matrix using coefficients D ---
    # Equation for cell i: diag[i]*Ti_new - (Flux_out[i+1] - Flux_in[i]) = diag[i]*T_star[i] + BC_Flux
    # Flux_out[i+1] = D_p[i] * (Ti+1_new - Ti_new) -> implicit
    # Flux_in[i]    = D_p[i-1] * (Ti_new - Tim1_new) -> implicit

    # Lower diagonal a[i] = coefficient of T[i-1] = -D_p[i-1] (for i=1 to Ng-1)
    if Ng >= 2:
        a[1:] = -diffusion_coeff_p[:-1]

    # Upper diagonal c[i] = coefficient of T[i+1] = -D_p[i] (for i=0 to Ng-2)
    if Ng >= 2:
        c[:-1] = -diffusion_coeff_p[:-1]

    # Main diagonal b[i] = diag[i] + D_p[i-1] (in-flux coeff) + D_p[i] (out-flux coeff)
    b[:] = diag_coeff[:]
    if Ng >= 2:
        b[:-1] += diffusion_coeff_p[:-1] # Add term from flux at face i+1
        b[1:] += diffusion_coeff_p[:-1] # Add term from flux at face i
    elif Ng == 1: # Single cell case
         b[0] += diffusion_coeff_p[0] # Only flux out at face 1 (rmax, coeff is 0)

    # --- Right Hand Side (RHS) Vector d ---
    # Start with previous value adjusted by advection (T_star)
    d[:] = diag_coeff[:] * T_g_star[:]

    # Add boundary condition flux terms
    # Face 0 (r=R): Heat flux INTO gas cell 0 is -q_gas_out_surf * A_g_faces[0]; outword direction is positive.
    if Ng >= 1:
        d[0] = diag_coeff[0] * T_g_star[0] + (-q_gas_out_surf) * A_g_faces[0]

    # Face Ng (r=rmax): Flux is zero, no modification needed for d[Ng-1]

    # --- Check for NaN ---
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d)):
        print("ERROR: NaN detected in gas diffusion matrix T build.")
        return None, None, None, None

    return a, b, c, d


def build_diffusion_matrix_gas_Y(
    k: int, # Species index
    Y_g_star: np.ndarray, # Species array after advection step [nsp, Ng]
    rho_g: np.ndarray, Dk_g: np.ndarray, # Props at T_old [Ng], [nsp, Ng]
    r_g_centers: np.ndarray, r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, nsp: int, dt: float,
    SpeciesFlux_face0_k: float, # Total species flux rate OUT of gas cell 0 at r=R [kg/s]
    A_g_faces: np.ndarray
    ):
    """Builds tridiagonal matrix for implicit species diffusion (Backward Euler)."""
    if Ng <= 0: return None, None, None, None
    a, b, c = np.zeros(Ng), np.zeros(Ng), np.zeros(Ng)
    d = np.zeros(Ng) # RHS vector

    Dk_g_k = Dk_g[k, :] # Diffusion coeff for species k at cell centers

    # Check for NaN/Inf in inputs
    if np.any(np.isnan(rho_g)) or np.any(np.isnan(Dk_g_k)) or np.any(np.isnan(Y_g_star[k,:])):
        print(f"ERROR: NaN input detected in build_diffusion_matrix_gas_Y for k={k}")
        return None, None, None, None

    # Coefficient for Y_new[i,k] contribution to its own cell balance
    diag_coeff = rho_g * volumes_g / dt

    # Calculate diffusion coefficient term D = rho * Dk * A / dr for faces i+1 = 1 to Ng
    if Ng >= 1:
        diffusion_coeff_p = np.zeros(Ng) # Coeff related to face i+1 (length Ng)

        # Internal faces (i+1 = 1 to Ng-1)
        if Ng >= 2:
            # Need rho * Dk at faces - use harmonic mean for Dk, arithmetic for rho?
            rhoDk_L = rho_g[:-1] * Dk_g_k[:-1]; rhoDk_R = rho_g[1:] * Dk_g_k[1:]
            rhoDk_f_internal = harmonic_mean(rhoDk_L, rhoDk_R) # Faces 1 to Ng-1
            dr_c_internal = r_g_centers[1:] - r_g_centers[:-1]
            dr_c_internal = np.maximum(dr_c_internal, 1e-15)
            diffusion_coeff_p[:-1] = rhoDk_f_internal * A_g_faces[1:Ng] / dr_c_internal

        # Outer face (i+1 = Ng) - Flux is zero (Neumann dYk/dr=0 -> J=0)
        diffusion_coeff_p[Ng-1] = 0.0

        diffusion_coeff_p = np.nan_to_num(diffusion_coeff_p)

    # --- Assemble the matrix ---
    # Eq: (diag[i] + D_p[i-1] + D_p[i]) * Yk_new[i] - D_p[i-1] * Yk_new[i-1] - D_p[i] * Yk_new[i+1] = diag[i]*Yk_star[i] + BC_Flux

    # Lower diagonal a[i] = -D_p[i-1] (for i=1 to Ng-1)
    if Ng >= 2: a[1:] = -diffusion_coeff_p[:-1]
    # Upper diagonal c[i] = -D_p[i] (for i=0 to Ng-2)
    if Ng >= 2: c[:-1] = -diffusion_coeff_p[:-1]
    # Main diagonal b[i] = diag[i] + D_p[i-1] + D_p[i]
    b[:] = diag_coeff[:]
    if Ng >= 2:
        b[:-1] += diffusion_coeff_p[:-1] # Term from face i+1
        b[1:] += diffusion_coeff_p[:-1] # Term from face i
    elif Ng == 1:
        b[0] += diffusion_coeff_p[0] # Only flux out at face 1 (rmax, coeff is 0)

    # --- Right Hand Side (RHS) Vector d ---
    # Start with previous value adjusted by advection (Y_star)
    d[:] = diag_coeff[:] * Y_g_star[k, :]

    # Add boundary condition flux terms
    # Face 0 (r=R): SpeciesFlux_face0_k is the rate OUT of cell 0 [kg/s]
    # Flux INTO cell 0 is -SpeciesFlux_face0_k. Add this to RHS.
    if Ng >= 1:
        d[0] -= SpeciesFlux_face0_k # Subtract flux out term from RHS

    # Face Ng (r=rmax): Flux is zero, no modification needed

    # --- Check for NaN ---
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d)):
        print(f"ERROR: NaN detected in gas diffusion matrix Y build for k={k}.")
        return None, None, None, None

    return a, b, c, d

# Note: calculate_gas_rhs_fvm is no longer used directly for time integration,
# but its internal logic for calculating fluxes and source terms might be reused
# or adapted for the advection and diffusion steps.
# For simplicity, let's keep it for now as a reference or for potential explicit steps.
