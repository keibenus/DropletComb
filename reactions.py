# reactions.py
"""Functions for calculating chemical reaction rates with cutoff option."""

import numpy as np
import cantera as ct
import config # Import config to access cutoff settings

def calculate_cantera_rates(gas_obj: ct.Solution, T_g: np.ndarray, P: float, Y_g: np.ndarray):
    """
    Calculates reaction rates using Cantera for detailed kinetics.
    Includes optional cutoff based on temperature and fuel mole fraction.
    """
    Ng = T_g.shape[0]
    nsp = gas_obj.n_species
    wdot = np.zeros((nsp, Ng))
    try:
        fuel_idx = gas_obj.species_index(config.FUEL_SPECIES_NAME)
    except ValueError:
        # This should ideally not happen if the mechanism contains the fuel
        print(f"CRITICAL ERROR: Fuel species '{config.FUEL_SPECIES_NAME}' not found in mechanism.")
        fuel_idx = -1 # Set invalid index

    # Temporary array for mole fractions if needed for cutoff
    X_g_i = np.zeros(nsp)

    points_reacted = 0 # Counter for debug
    for i in range(Ng):
        calculate_reactions = True
        fuel_mole_frac = -1.0 # Initialize
        # --- Reaction Cutoff Check ---
        if config.ENABLE_REACTION_CUTOFF and fuel_idx != -1:
            # Check basic conditions first
            if T_g[i] < config.REACTION_CALC_MIN_TEMP or np.isnan(T_g[i]) or np.isnan(P):
                 calculate_reactions = False
            else:
                 # Check Y_g validity before setting state
                 Y_g_i_check = Y_g[:, i].copy()
                 Y_g_i_check = np.maximum(Y_g_i_check, 0)
                 sum_Y_check = np.sum(Y_g_i_check)
                 if abs(sum_Y_check - 1.0) > 1e-4 and sum_Y_check > 1e-6: Y_g_i_check /= sum_Y_check
                 elif sum_Y_check <= 1e-6: Y_g_i_check *= 0.0

                 try:
                     # Set state to get mole fractions for cutoff check
                     gas_obj.TPY = T_g[i], P, Y_g_i_check
                     X_g_i = gas_obj.X
                     fuel_mole_frac = X_g_i[fuel_idx]
                 except (ct.CanteraError, ValueError):
                      # If state cannot be set, assume no reaction for safety
                      fuel_mole_frac = -1.0
                      # print(f"Warning: Could not set state for reaction cutoff check at node {i}")

                 if fuel_mole_frac < config.REACTION_CALC_MIN_FUEL_MOL_FRAC:
                     calculate_reactions = False
        # --- End Cutoff Check ---

        if calculate_reactions:
            points_reacted += 1
            try:
                # Ensure mass fractions are valid before getting rates
                Y_g_i = Y_g[:, i].copy() # Use a copy
                sum_Y = np.sum(Y_g_i)
                if abs(sum_Y - 1.0) > 1e-4:
                    Y_g_i = np.maximum(Y_g_i, 0)
                    if sum_Y > 1e-6: Y_g_i /= sum_Y
                    else: Y_g_i *= 0.0

                # Set state and get rates - Use safe set_state? Assume direct TPY is ok if checked above
                gas_obj.TPY = T_g[i], P, Y_g_i
                wdot[:, i] = gas_obj.net_production_rates * gas_obj.molecular_weights # kg/m^3/s
            except (ct.CanteraError, ValueError) as e:
                 # print(f"Warning: Cantera failed during rate calc at point {i}. Rates set to 0. T={T_g[i]:.1f}, P={P:.1e}") # Reduce verbosity
                 wdot[:, i] = 0.0
        else:
            # Set rates to zero if cutoff conditions met
            wdot[:, i] = 0.0

    # Log how many points had reactions calculated (for debugging cutoff)
    if config.LOG_LEVEL >= 2:
        print(f"      DEBUG Reactions: Calculated for {points_reacted}/{Ng} points.", end='\r', flush=True)

    return wdot


def calculate_overall_rates(rho_g: np.ndarray, T_g: np.ndarray, Y_g: np.ndarray, gas_obj: ct.Solution, P: float): # Added P
    """
    Calculates reaction rates using the specified overall Arrhenius model.
    Includes optional cutoff based on temperature and fuel mole fraction.
    NOTE: Now implemented with looping for cutoff check.
    """
    Ng = T_g.shape[0]
    nsp = gas_obj.n_species
    wdot_mass = np.zeros((nsp, Ng))

    try:
        fuel_idx = gas_obj.species_index(config.OVERALL_FUEL)
        ox_idx = gas_obj.species_index(config.OVERALL_OXIDIZER)
        products = {'CO2': (gas_obj.species_index('CO2'), 7.0),
                    'H2O': (gas_obj.species_index('H2O'), 8.0)}
        nu_fuel = -1.0; nu_ox = -11.0 # Stoichiometry for n-heptane
    except ValueError as e:
        print(f"CRITICAL ERROR finding species indices for overall reaction: {e}")
        return wdot_mass # Return zeros

    Mf = gas_obj.molecular_weights[fuel_idx]
    Mo = gas_obj.molecular_weights[ox_idx]
    Mp = {name: gas_obj.molecular_weights[idx] for name, (idx, nu) in products.items()}
    X_g_i = np.zeros(nsp) # Temp array for mole fractions
    points_reacted = 0

    for i in range(Ng):
        calculate_reactions = True
        fuel_mole_frac = -1.0
        # --- Reaction Cutoff Check ---
        if config.ENABLE_REACTION_CUTOFF:
             if T_g[i] < config.REACTION_CALC_MIN_TEMP or np.isnan(T_g[i]) or np.isnan(P):
                 calculate_reactions = False
             else:
                 Y_g_i_check = Y_g[:, i].copy(); Y_g_i_check = np.maximum(Y_g_i_check, 0)
                 sum_Y_check = np.sum(Y_g_i_check)
                 if abs(sum_Y_check - 1.0) > 1e-4 and sum_Y_check > 1e-6: Y_g_i_check /= sum_Y_check
                 elif sum_Y_check <= 1e-6: Y_g_i_check *= 0.0
                 try:
                     gas_obj.TPY = T_g[i], P, Y_g_i_check # Use current P
                     X_g_i = gas_obj.X
                     fuel_mole_frac = X_g_i[fuel_idx]
                 except (ct.CanteraError, ValueError): fuel_mole_frac = -1.0

                 if fuel_mole_frac < config.REACTION_CALC_MIN_FUEL_MOL_FRAC:
                     calculate_reactions = False
        # --- End Cutoff Check ---

        if calculate_reactions:
            Yf_i = Y_g[fuel_idx, i]
            Yo_i = Y_g[ox_idx, i]
            T_i = T_g[i]
            rho_i = rho_g[i]
            T_threshold = 298.0; Y_threshold = 1e-10 # Rate calc thresholds

            if T_i > T_threshold and Yf_i > Y_threshold and Yo_i > Y_threshold and rho_i > 1e-6:
                points_reacted += 1
                try:
                    exp_term = np.exp(-config.OVERALL_E_SI / (config.R_UNIVERSAL * T_i))
                    omega_mol_fuel_i = (config.OVERALL_B_SI * rho_i**2 *
                                        Yf_i * Yo_i / (Mf * Mo) * exp_term)
                    # Convert to mass rates [kg/m^3/s]
                    wdot_mass[fuel_idx, i] = omega_mol_fuel_i * nu_fuel * Mf
                    wdot_mass[ox_idx, i] = omega_mol_fuel_i * nu_ox * Mo
                    for name, (idx, nu_prod) in products.items():
                        wdot_mass[idx, i] = omega_mol_fuel_i * nu_prod * Mp[name]
                except OverflowError:
                    print(f"Warning: Overflow calculating overall rate at T={T_i:.1f}K. Setting rate to 0.")
                    wdot_mass[:, i] = 0.0 # Set rate to 0 if exp overflows
            else:
                wdot_mass[:, i] = 0.0 # No reaction if below thresholds
        else:
             wdot_mass[:, i] = 0.0 # No reaction if cutoff

    if config.LOG_LEVEL >= 2:
        print(f"      DEBUG Overall Reactions: Calculated for {points_reacted}/{Ng} points.", end='\r', flush=True)

    return wdot_mass