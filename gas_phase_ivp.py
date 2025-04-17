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