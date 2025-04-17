# =========================================
#        interface.py (修正後)
# =========================================
# interface.py
"""Functions for calculating conditions at the liquid-gas interface (FVM context)."""

import numpy as np
import cantera as ct
import config
from properties import LiquidProperties, GasProperties # For type hinting
from numerics import gradient_at_face, harmonic_mean, arithmetic_mean # Import necessary helpers
import scipy.optimize as op

def calculate_vapor_pressure(T_s, liquid_props: LiquidProperties):
    """Gets vapor pressure from liquid properties."""
    T_s_phys = max(200.0, min(T_s, 2000.0))
    return liquid_props.get_prop('vapor_pressure', T_s_phys) # Pa

def calculate_fuel_mass_fraction_surf(T_s, P, gas_props: GasProperties, liquid_props: LiquidProperties):
    """
    Calculates fuel mass fraction and equilibrium composition array
    at the surface assuming phase equilibrium.
    """
    P_sat = calculate_vapor_pressure(T_s, liquid_props)
    P_sat = min(max(P_sat, 0.0), P * 0.9999) # Ensure 0 <= P_sat < P

    X_f_surf = P_sat / P if P > 1e-6 else 0.0 # Mole fraction
    X_f_surf = np.clip(X_f_surf, 0.0, 1.0) # Ensure physical bounds

    gas = gas_props.gas
    fuel_idx = gas_props.fuel_idx
    nsp = gas_props.nsp
    mw = gas_props.molecular_weights

    X_surf = np.zeros(nsp)
    if fuel_idx >= 0: X_surf[fuel_idx] = X_f_surf
    else: return 0.0, np.zeros(nsp) # Should not happen if config is correct

    X_non_fuel_total = 1.0 - X_f_surf
    if X_non_fuel_total < 0.0: X_non_fuel_total = 0.0

    X_amb_non_fuel = {}
    total_X_amb_non_fuel = 0.0
    for sp_name, x_val in config.X_INF_INIT.items():
        if sp_name != config.FUEL_SPECIES_NAME:
            try:
                 sp_idx = gas_props.gas.species_index(sp_name)
                 X_amb_non_fuel[sp_name] = x_val
                 total_X_amb_non_fuel += x_val
            except ValueError: pass # Ignore species not in mech

    if total_X_amb_non_fuel > 1e-6:
        for i in range(nsp):
            if i != fuel_idx:
                 sp_name = gas_props.species_names[i]
                 ambient_frac = X_amb_non_fuel.get(sp_name, 0.0)
                 X_surf[i] = X_non_fuel_total * (ambient_frac / total_X_amb_non_fuel)

    X_surf = np.maximum(X_surf, 0)
    sum_X = np.sum(X_surf)
    if sum_X > 1e-9: X_surf /= sum_X
    else:
        X_surf.fill(0.0)
        if gas_props.n2_idx >= 0: X_surf[gas_props.n2_idx] = 1.0
        else: X_surf[0] = 1.0

    mean_mw_surf = np.sum(X_surf * mw) # kg/kmol
    Y_surf = np.zeros(nsp)
    if mean_mw_surf > 1e-6:
         Y_surf = X_surf * mw / mean_mw_surf # kg/kg
         Y_surf = np.maximum(Y_surf, 0)
         sum_Y = np.sum(Y_surf)
         if abs(sum_Y - 1.0) > 1e-6 and sum_Y > 1e-9: Y_surf /= sum_Y
         elif sum_Y <= 1e-9:
             Y_surf.fill(0.0)
             if gas_props.n2_idx >= 0: Y_surf[gas_props.n2_idx] = 1.0
             else: Y_surf[0] = 1.0

    Y_f_surf_equil = Y_surf[fuel_idx] if fuel_idx >=0 else 0.0

    # Check: Y_f_eq should be small at low T_s, high P
    if T_s < 350 and Y_f_surf_equil > 0.1 and config.LOG_LEVEL >= 1:
         print(f"DEBUG CHECK: Y_f_eq={Y_f_surf_equil:.4f} seems high for T_s={T_s:.1f}K, Psat={P_sat:.2e}Pa, P={P:.2e}Pa, Xf={X_f_surf:.4f}")

    return Y_f_surf_equil, Y_surf

def _interface_energy_residual(T_s_guess: float,
                              T_l_node_last: float, T_g_node0: float,
                              Y_g_node0: np.ndarray, P: float, R: float,
                              r_l_node_last_center: float, r_g_node0_center: float,
                              gas_props: GasProperties, liquid_props: LiquidProperties, Nl: int):
    """
    Calculates the energy balance residual at the interface for a guessed Ts.
    Residual = Heat_In_Gas + Heat_In_Liq - Heat_Evap
    """
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx

    # Avoid non-physical temperatures during iteration
    T_s_guess = max(200.0, min(T_s_guess, T_g_node0 + 100)) # Limit guess range

    # --- 1. Equilibrium composition and properties at surface (T=Ts_guess) ---
    try:
        Y_f_eq, Y_eq = calculate_fuel_mass_fraction_surf(T_s_guess, P, gas_props, liquid_props)

        props_l_s = liquid_props.get_properties(T_s_guess)
        lambda_l_s = props_l_s.get('thermal_conductivity', 1e-3)
        Lv = props_l_s.get('heat_of_vaporization', 1e6)
        rho_l_s = props_l_s.get('density', 100.0) # Needed only if mdot calc fails?

        if not gas_props.set_state(T_s_guess, P, Y_eq):
            # If state setting fails at guess, use cell 0 props as fallback for surface
            if not gas_props.set_state(T_g_node0, P, Y_g_node0): return 1e6 # Return large error
            lambda_g_s = gas_props.gas.thermal_conductivity
            rho_g_s = gas_props.get_density(T_g_node0, P, Y_g_node0)
            Dk_s = gas_props.get_diffusion_coeffs(T_g_node0, P, Y_g_node0)
        else:
            lambda_g_s = gas_props.gas.thermal_conductivity
            rho_g_s = gas_props.get_density(T_s_guess, P, Y_eq)
            Dk_s = gas_props.get_diffusion_coeffs(T_s_guess, P, Y_eq)

        # --- 2. Properties at center of first gas cell (i=0) ---
        rho_g_0 = gas_props.get_density(T_g_node0, P, Y_g_node0)
        if not gas_props.set_state(T_g_node0, P, Y_g_node0):
            lambda_g_0 = lambda_g_s; Dk_0 = Dk_s # Fallback
        else:
            lambda_g_0 = gas_props.gas.thermal_conductivity
            Dk_0 = gas_props.get_diffusion_coeffs(T_g_node0, P, Y_g_node0)

        # --- 3. Estimate Properties at the Interface Face (r=R) ---
        lambda_g_face = harmonic_mean(lambda_g_s, lambda_g_0)
        rho_g_face = arithmetic_mean(rho_g_s, rho_g_0)
        Dk_face = harmonic_mean(Dk_s, Dk_0)

        # --- 4. Calculate Gradients at the Interface Face (r=R) ---
        dr_face_center_g = r_g_node0_center - R
        if dr_face_center_g < 1e-15: grad_Tg_face = 0.0; grad_Yk_face = np.zeros(nsp)
        else: grad_Tg_face = (T_g_node0 - T_s_guess) / dr_face_center_g; grad_Yk_face = (Y_g_node0 - Y_eq) / dr_face_center_g

        if Nl <= 1: grad_Tl_face = 0.0
        else:
            dr_face_center_l = R - r_l_node_last_center
            if dr_face_center_l < 1e-15: grad_Tl_face = 0.0
            else: grad_Tl_face = (T_s_guess - T_l_node_last) / dr_face_center_l

        # --- 5. Calculate Heat Fluxes TOWARDS Interface [W/m^2] ---
        q_gas_out_at_surf = -lambda_g_face * grad_Tg_face # Should be negative initially
        q_liq_out_at_surf = -lambda_l_s * grad_Tl_face # Should be negative initially

        # --- 6. Calculate Mass Flux (mdot'') using Stefan Flow Condition ---
        fuel_idx = gas_props.fuel_idx
        denominator = 1.0 - Y_f_eq
        # Diffusive flux OUTWARDS at the face
        diff_flux_f_outwards = - rho_g_face * Dk_face[fuel_idx] * grad_Yk_face[fuel_idx]
        if denominator > 1e-6:
            mdot_double_prime = diff_flux_f_outwards / denominator
        else: # Fallback to energy balance ONLY if Yf is actually high
            if Y_f_eq > 0.999:
                if config.LOG_LEVEL >= 1: print(f"Warning: Stefan denom near zero (Yfeq={Y_f_eq:.4f}). Using Energy Balance mdot''.")
                # Energy Balance: Heat_In_Gas + Heat_In_Liq = mdot * Lv
                # Heat_In_Gas = -q_gas_out_at_surf
                # Heat_In_Liq = -q_liq_out_at_surf
                mdot_double_prime = max(0.0, (-q_gas_out_at_surf - q_liq_out_at_surf) / max(Lv, 1e3))
            else:
                if config.LOG_LEVEL >= 1: print(f" Warning: Stefan denominator small but Yfeq low ({Y_f_eq:.3f}). Check grads/props. Setting mdot=0.")
                mdot_double_prime = 0.0

        mdot_double_prime = max(0.0, mdot_double_prime)
        # --- 7. Calculate Residual ---
        # Energy balance: q_gas_in + q_liq_in = mdot * Lv
        # q_gas_in = -q_gas_out_at_surf
        # q_liq_in = -q_liq_out_at_surf
        residual = (-q_gas_out_at_surf) + (-q_liq_out_at_surf) - (mdot_double_prime * Lv)

    except Exception as e_res:
        print(f"ERROR in residual function for T_s={T_s_guess:.2f}K: {e_res}")
        residual = 1e10 # Return large error if calculation fails

    # print(f"Debug Residual: Ts={T_s_guess:.2f}, Res={residual:.3e}, qg={q_gas_to_surf:.3e}, ql={q_liq_to_surf:.3e}, mdot={mdot_double_prime:.3e}")
    return residual


def solve_interface_conditions(
    T_l_node_last: float, T_g_node0: float,
    Y_g_node0: np.ndarray, P: float, R: float,
    r_l_node_last_center: float, r_g_node0_center: float,
    gas_props: GasProperties, liquid_props: LiquidProperties, Nl: int
    ):
    """
    Solves for the interface temperature (Ts) that satisfies the energy balance,
    and returns all consistent interface quantities.
    """
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx
    # --- Find Ts by solving residual(Ts) = 0 ---
    # Define bounds for the solver [T_liquid_side, T_gas_side]
    # Add some buffer, but avoid crossing other cell temperatures drastically?
    T_lower_bound = max(150.0, T_l_node_last - 20.0) # Lower bound slightly below T_l
    T_upper_bound = min(5000.0, T_g_node0 + 100.0) # Upper bound slightly above T_g

    # Ensure lower < upper
    if T_lower_bound >= T_upper_bound:
         T_lower_bound = T_upper_bound - 1.0 # Create small interval if needed

    # Initial guess (e.g., average or liquid temp)
    # T_s_guess_initial = 0.5 * (T_l_node_last + T_g_node0)
    T_s_guess_initial = T_l_node_last # Start with liquid temp

    T_s_solution = T_s_guess_initial # Default if solver fails
    solved = False
    try:
        # Check if residual function changes sign within bounds
        res_low = _interface_energy_residual(T_lower_bound, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl)
        res_high = _interface_energy_residual(T_upper_bound, T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl)

        if np.sign(res_low) != np.sign(res_high):
            T_s_solution, r_info = op.brentq(
                _interface_energy_residual,
                a=T_lower_bound, b=T_upper_bound,
                args=(T_l_node_last, T_g_node0, Y_g_node0, P, R, r_l_node_last_center, r_g_node0_center, gas_props, liquid_props, Nl),
                xtol=1e-3, rtol=1e-3, # Tolerances for Ts convergence
                maxiter=30, full_output=True
            )
            if r_info.converged:
                solved = True
            else:
                 if config.LOG_LEVEL >= 1: print(f"Warning: Interface Ts solver (brentq) did not converge. Flag: {r_info.flag}")
        else:
            # Sign did not change, root likely not bracketed or residual is flat (e.g., mdot=0, q=0)
            if config.LOG_LEVEL >= 1: print(f"Warning: Interface Ts root not bracketed [{T_lower_bound:.1f}, {T_upper_bound:.1f}]. Res: [{res_low:.2e}, {res_high:.2e}]. Using initial guess.")
            # Check if residual is already close to zero at bounds?
            if abs(res_low) < 1e-1: T_s_solution = T_lower_bound; solved = True
            elif abs(res_high) < 1e-1: T_s_solution = T_upper_bound; solved = True
            else: T_s_solution = T_l_node_last # Fallback to liquid temperature

    except ValueError as e_solve: # Handle potential errors from brentq (e.g., bounds)
        print(f"ERROR during interface Ts solve: {e_solve}. Using T_l_node_last.")
        T_s_solution = T_l_node_last # Use liquid temperature as fallback
    except Exception as e_gen:
         print(f"Unexpected ERROR during interface Ts solve: {e_gen}")
         T_s_solution = T_l_node_last

    # Clip final solution to physical bounds
    T_s_final = np.clip(T_s_solution, T_lower_bound-10, T_upper_bound+10) # Allow slight over/undershoot
    #T_s_final = 301
    #print(f"T_s_final={T_s_final}")
    

    # --- Recalculate final fluxes with converged Ts ---
    # Call residual one last time to get consistent fluxes based on T_s_final
    # Need to slightly modify residual to return values or recalculate here...
    # Let's recalculate here for clarity
    Y_f_eq_final, Y_eq_final = calculate_fuel_mass_fraction_surf(T_s_final, P, gas_props, liquid_props)
    props_l_s = liquid_props.get_properties(T_s_final)
    lambda_l_s = props_l_s.get('thermal_conductivity', 1e-3)
    Lv = props_l_s.get('heat_of_vaporization', 1e6)
    rho_l_s = props_l_s.get('density', 100.0)

    if not gas_props.set_state(T_s_final, P, Y_eq_final): gas_props.set_state(T_g_node0, P, Y_g_node0) # Fallback
    lambda_g_s = gas_props.gas.thermal_conductivity
    rho_g_s = gas_props.get_density(T_s_final, P, Y_eq_final)
    Dk_s = gas_props.get_diffusion_coeffs(T_s_final, P, Y_eq_final)

    if not gas_props.set_state(T_g_node0, P, Y_g_node0): lambda_g_0=lambda_g_s; Dk_0=Dk_s # Fallback
    else: lambda_g_0 = gas_props.gas.thermal_conductivity; Dk_0 = gas_props.get_diffusion_coeffs(T_g_node0, P, Y_g_node0)
    rho_g_0 = gas_props.get_density(T_g_node0, P, Y_g_node0)

    lambda_g_face = harmonic_mean(lambda_g_s, lambda_g_0)
    rho_g_face = arithmetic_mean(rho_g_s, rho_g_0)
    Dk_face = harmonic_mean(Dk_s, Dk_0)
    #print(f"Dk_s={Dk_s} Dk_0={Dk_0}")

    dr_face_center_g = r_g_node0_center - R
    #print(f"dr_face_center_g={dr_face_center_g} r_g_node0_center={r_g_node0_center} R={R}")
    if dr_face_center_g < 1e-15: grad_Tg_face = 0.0; grad_Yk_face = np.zeros(nsp)
    else: grad_Tg_face = (T_g_node0 - T_s_final) / dr_face_center_g; grad_Yk_face = (Y_g_node0 - Y_eq_final) / dr_face_center_g

    if Nl <= 1: grad_Tl_face = 0.0
    else:
        dr_face_center_l = R - r_l_node_last_center
        if dr_face_center_l < 1e-15: grad_Tl_face = 0.0
        else: grad_Tl_face = (T_s_final - T_l_node_last) / dr_face_center_l
    #print(f"NL={Nl} R={R} r_l_node_last_center={r_l_node_last_center} T_s_final={T_s_final} T_l_node_last={T_l_node_last}")

    # Final mdot'' calculation (Stefan preferred)
    Y_f_eq_final = Y_eq_final[fuel_idx]
    denominator = 1.0 - Y_f_eq_final
    diff_flux_f_outwards = - rho_g_face * Dk_face[fuel_idx] * grad_Yk_face[fuel_idx]
    if denominator > 1e-6: mdot = diff_flux_f_outwards / denominator
    else: # Use energy balance as fallback (less preferred now Ts is solved)
        q_gas_out = -lambda_g_face * grad_Tg_face
        q_liq_out = -lambda_l_s * grad_Tl_face
        mdot = max(0.0, (-q_gas_out - q_liq_out) / max(Lv, 1e3))
    mdot = max(0.0, mdot)

    # Final heat fluxes (Positive Outwards)
    q_gas_out_surf = -lambda_g_face * grad_Tg_face
    q_liq_out_surf = -lambda_l_s * grad_Tl_face

    # Final surface velocity
    rho_g_s = gas_props.get_density(T_s_final, P, Y_eq_final) # Recalculate rho at final Ts
    v_g_surf = mdot / rho_g_s if rho_g_s > 1e-9 else 0.0

    # Return fluxes CONSISTENTLY (e.g., always return flux INTO the phase from interface)
    # q_gas_to_surf = -q_gas_out_surf # Positive if heat flows from gas to interface
    # q_liq_to_surf = -q_liq_out_surf # Positive if heat flows from liquid to interface
    # Let's return the OUTWARD fluxes and handle signs in calling functions
    # return mdot, q_gas_out_surf, q_liq_out_surf, v_g_surf, Y_eq_final, T_s_final

    # OR return fluxes TOWARDS interface for clarity in energy balance check?
    #q_gas_to_surf = -q_gas_out_surf
    #q_liq_to_surf = -q_liq_out_surf
    return mdot, q_gas_out_surf, q_liq_out_surf, v_g_surf, Y_eq_final, T_s_final

####### now not use ########
def calculate_interface_fluxes(
    T_s: float,            # Liquid surface temperature (K)
    T_g_node0: float,      # Gas temperature at center of cell 0 [K]
    Y_g_node0: np.ndarray, # Gas composition at center of cell 0 [nsp]
    P: float,              # Pressure (Pa)
    R: float,              # Current radius (m)
    r_g_node0_center: float, # Radius of gas cell 0 center [m]
    T_l_node_last: float,  # Liquid temperature at center of last cell (Nl-1)
    r_l_node_last_center: float, # Radius of liquid cell Nl-1 center
    gas_props: GasProperties,
    liquid_props: LiquidProperties,
    Nl: int # Pass Nl to handle Nl=1 case
    ):
    """
    Calculates key interface quantities (mdot'', heat fluxes, surface velocity)
    for FVM boundary conditions. Uses FVM principles. Prioritizes Stefan flux for mdot''.
    Returns:
        mdot_double_prime: Mass flux away from surface [kg/m^2/s] >= 0
        q_gas_to_surf: Heat flux from gas TO surface [W/m^2] >= 0 if Tg > Ts
        q_liq_to_surf: Heat flux from liquid TO surface [W/m^2] >= 0 if Tl_in > Ts
        v_g_surf: Gas velocity normal TO surface [m/s] >= 0
        Y_equilibrium: Equilibrium mass fractions at surface [-]
    """
    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx

    # --- 1. Equilibrium composition and properties at surface (T=Ts) ---
    Y_f_equilibrium, Y_equilibrium = calculate_fuel_mass_fraction_surf(T_s, P, gas_props, liquid_props)

    props_l_s = liquid_props.get_properties(T_s)
    lambda_l_s = props_l_s.get('thermal_conductivity', 1e-3)
    Lv = props_l_s.get('heat_of_vaporization', 1e6)
    rho_l_s = props_l_s.get('density', 100.0)

    # Get gas properties at surface state Ts, Yeq
    if not gas_props.set_state(T_s, P, Y_equilibrium):
        print(f"Warning: Failed set gas state at surface T_s={T_s:.1f}K. Using cell 0 props.")
        # Fallback: Use cell 0 state for surface gas properties
        if not gas_props.set_state(T_g_node0, P, Y_g_node0):
            print("ERROR: Failed to set state for cell 0 either. Cannot get surf props.")
            return 0.0, 0.0, 0.0, 0.0, Y_equilibrium
    lambda_g_s = gas_props.gas.thermal_conductivity
    rho_g_s = gas_props.get_density(gas_props.gas.T, P, gas_props.gas.Y)
    Dk_s = gas_props.get_diffusion_coeffs(gas_props.gas.T, P, gas_props.gas.Y, option=config.DIFFUSION_OPTION)

    # --- 2. Properties at center of first gas cell (i=0) ---
    rho_g_0 = gas_props.get_density(T_g_node0, P, Y_g_node0) # Already calculated potentially, recalc ensures consistency
    if not gas_props.set_state(T_g_node0, P, Y_g_node0): # Set state for other props
        print(f"Warning: Failed set state gas cell 0 T={T_g_node0:.1f}K. Using surface props.")
        lambda_g_0 = lambda_g_s; Dk_0 = Dk_s
    else:
        lambda_g_0 = gas_props.gas.thermal_conductivity
        Dk_0 = gas_props.get_diffusion_coeffs(T_g_node0, P, Y_g_node0, option=config.DIFFUSION_OPTION)

    # --- 3. Estimate Properties at the Interface Face (r=R) ---
    lambda_g_face = harmonic_mean(lambda_g_s, lambda_g_0)
    rho_g_face = arithmetic_mean(rho_g_s, rho_g_0)
    Dk_face = harmonic_mean(Dk_s, Dk_0)

    # Handle potential NaN/invalid values
    prop_list = [lambda_l_s, Lv, lambda_g_face, rho_g_face, rho_l_s]
    prop_list.extend(Dk_face)
    if np.any(np.isnan(prop_list)) or Lv < 1e-3 or rho_l_s < 1e-3 or rho_g_face < 1e-3:
        print(f"Warning: Invalid props at interface face (T_s={T_s:.1f}K). Fluxes=0.")
        return 0.0, 0.0, 0.0, 0.0, Y_equilibrium

    # --- 4. Calculate Gradients at the Interface Face (r=R) ---
    dr_face_center_g = r_g_node0_center - R
    if dr_face_center_g < 1e-15: # Use a small epsilon
        print("Warning: Gas cell 0 too thin for gradient calc.")
        grad_Tg_face = 0.0; grad_Yk_face = np.zeros(nsp)
    else:
        grad_Tg_face = (T_g_node0 - T_s) / dr_face_center_g
        grad_Yk_face = (Y_g_node0 - Y_equilibrium) / dr_face_center_g

    if Nl <= 1:
        grad_Tl_face = 0.0 # No internal gradient for single liquid cell
    else:
        dr_face_center_l = R - r_l_node_last_center
        if dr_face_center_l < 1e-15:
             print("Warning: Liquid cell Nl-1 too thin for gradient calc.")
             grad_Tl_face = 0.0
        else:
             grad_Tl_face = (T_s - T_l_node_last) / dr_face_center_l

    # --- 5. Calculate Mass Flux (mdot'') using Stefan Flow Condition ---
    # mdot'' = (-rho_face * D_f_face * dYf/dr|_face) / (1 - Y_f_equilibrium)
    denominator = 1.0 - Y_f_equilibrium
    # Calculate diffusive flux of fuel TOWARDS the interface
    diff_flux_f_to_surf = - rho_g_face * Dk_face[fuel_idx] * grad_Yk_face[fuel_idx]
    print(f"rho_g_face={rho_g_face} Dk_face={Dk_face[fuel_idx]} grad_Yk_face={grad_Yk_face[fuel_idx]}")

    # Avoid division by zero/small numbers ONLY if Y_f_eq is truly close to 1
    # Check if Y_f_eq is unexpectedly high (potential issue)
    if denominator < 1e-4: # Use a threshold like 0.9999 for Y_f_eq
        if config.LOG_LEVEL >= 1: print(f"Warning: Stefan denom near zero (Yfeq={Y_f_equilibrium:.4f}) at T_s={T_s:.1f}K. Check Y_f_eq calc. Using Energy Balance mdot''.")
        # Fallback calculation using Energy Balance (fluxes defined positive towards interface)
        q_gas_to_surf_E = lambda_g_face * (-grad_Tg_face)
        q_liq_to_surf_E = lambda_l_s * (-grad_Tl_face)
        mdot_double_prime = max(0.0, (q_gas_to_surf_E + q_liq_to_surf_E) / Lv)
    else:
        # Use Stefan flux (flux away from surface is positive)
        mdot_double_prime = diff_flux_f_to_surf / denominator

    # Ensure physical evaporation rate
    mdot_double_prime = max(0.0, mdot_double_prime)

    # --- 6. Calculate Heat Fluxes TOWARDS Interface ---
    # q = -lambda * dT/dr (Positive flux = heat flow in -r direction)
    # Flux TOWARDS interface (in +r direction for liquid, -r direction for gas)
    ##q_gas_to_surf = lambda_g_face * (-grad_Tg_face) # Positive if gas is hotter
    ##q_liq_to_surf = lambda_l_s * (-grad_Tl_face) # Positive if liquid interior is hotter

    q_gas_to_surf = lambda_g_face * (T_g_node0 - T_s) / dr_face_center_g if dr_face_center_g > 1e-15 else 0.0
    q_liq_to_surf = lambda_l_s * (T_l_node_last - T_s) / dr_face_center_l if (Nl > 1 and dr_face_center_l > 1e-15) else 0.0

    print(f"lambda_g_face={lambda_g_face} grad_Tg_face={grad_Tg_face} q_gas_to_surf={q_gas_to_surf} T_g_node0={T_g_node0} T_s={T_s}")
    print(f"lambda_l_s={lambda_l_s} grad_Tl_face={grad_Tl_face} q_liq_to_surf={q_liq_to_surf} T_l_node_last={T_l_node_last} T_s={T_s}")

    ### Ensure fluxes are physically reasonable (non-negative if source is hotter)
    ##q_gas_to_surf = max(0.0, q_gas_to_surf) if T_g_node0 >= T_s else min(0.0, q_gas_to_surf)
    ##q_liq_to_surf = max(0.0, q_liq_to_surf) if T_l_node_last >= T_s else min(0.0, q_liq_to_surf)

    # --- Optional: Energy Balance Check (Re-calculate q_evap based on mdot'') ---
    q_evap = mdot_double_prime * Lv
    balance_error = (q_gas_to_surf + q_liq_to_surf) - q_evap
    if abs(balance_error) > 0.5 * (abs(q_gas_to_surf) + abs(q_liq_to_surf) + abs(q_evap) + 1e-3):
        if config.LOG_LEVEL >=1:
            # This warning might appear if Stefan mdot'' is used, as it relies on Y gradient
            # while q relies on T gradients. Indicate which mdot was used.
            mdot_method = "Stefan" if denominator >= 1e-4 else "EnergyBalance"
            print(f"Warning: Interface energy balance error ({mdot_method} mdot'') at T_s={T_s:.1f}K: {balance_error:.2e} W/m2")
            # print(f"  q_gas->s={q_gas_to_surf:.2e}, q_liq->s={q_liq_to_surf:.2e}, q_evap={q_evap:.2e} (mdot''={mdot_double_prime:.2e})")
            # print(f"  GradT_g={grad_Tg_face:.2e}, GradT_l={grad_Tl_face:.2e}, GradYf={grad_Yk_face[fuel_idx]:.2e}")

    # --- 7. Calculate Gas Velocity Normal TO Surface ---
    v_g_surf = mdot_double_prime / rho_g_s if rho_g_s > 1e-9 else 0.0 # Velocity away from surface

    return mdot_double_prime, q_gas_to_surf, q_liq_to_surf, v_g_surf, Y_equilibrium