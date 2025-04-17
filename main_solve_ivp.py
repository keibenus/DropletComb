# =========================================
#      main_solve_ivp.py (修正後)
# =========================================
# main_solve_ivp.py
"""
Main script for droplet evaporation and combustion simulation using FVM.
Uses scipy.integrate.solve_ivp with BDF method.
Includes LOG_LEVEL for controlling output verbosity.
Allows disabling reactions via config.REACTION_TYPE = 'none'.
"""

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Import simulation modules
import config # Import config first
import grid
import properties
import interface
import reactions
import numerics # Still needed for interpolation/averaging helpers
import liquid_phase # For liquid RHS (FVM version)
import gas_phase    # For gas RHS (FVM version)
import plotting

# Global variable for max dT/dt (or use a class structure)
max_dTdt_current = 0.0
ode_call_count = 0 # Counter for ODE calls

# --- Define ODE function for solve_ivp ---
def ode_system(t, y, gas_props: properties.GasProperties, liquid_props: properties.LiquidProperties):
    """
    Defines the system of ODEs dy/dt = f(t, y) for solve_ivp using FVM.
    Extra arguments (gas_props, liquid_props) are passed via solve_ivp's `args` parameter.
    """
    global max_dTdt_current, ode_call_count # Access global variables
    ode_call_count += 1
    if config.LOG_LEVEL >= 1:
        print(f"  ode_system call #{ode_call_count} at t = {t:.6e} s", end='\r', flush=True)

    # --- Unpack state vector y ---
    Nl = config.NL; Ng = config.NG; nsp = gas_props.nsp
    idx_Tl_end = Nl; idx_Tg_end = Nl + Ng; idx_Yg_end = Nl + Ng + nsp * Ng; idx_R = len(y) - 1
    T_l_centers = y[0:idx_Tl_end]
    T_g_centers = y[idx_Tl_end:idx_Tg_end]
    Y_g_flat = y[idx_Tg_end:idx_Yg_end]
    R_current = y[idx_R]

    # --- Basic State Validity Checks ---
    if R_current <= 1e-9:
        if config.LOG_LEVEL >= 1: print("Droplet radius near zero, returning zero derivatives.")
        return np.zeros_like(y)
    if np.any(T_l_centers < 100) or np.any(T_g_centers < 100):
        if config.LOG_LEVEL >= 0: print(f"Warning: Very low temperature detected T_l_min={np.min(T_l_centers):.1f}, T_g_min={np.min(T_g_centers):.1f}. Returning zero derivatives.")
        return np.zeros_like(y)
    if np.any(np.isnan(y)):
        print("ERROR: NaN detected in input state vector 'y'. Returning zero derivatives.")
        return np.zeros_like(y)

    # Reshape Y_g and ensure validity
    Y_g_centers = Y_g_flat.reshape((nsp, Ng))
    Y_g_centers = np.maximum(Y_g_centers, 0)
    sum_Y = np.sum(Y_g_centers, axis=0)
    if np.any(abs(sum_Y - 1.0) > 1e-6):
        mask = (sum_Y > 1e-9)
        Y_g_centers[:, mask] /= sum_Y[mask]
        if np.any(~mask):
            Y_g_centers[:, ~mask] = 0.0
            n2_idx = gas_props.n2_idx
            if n2_idx >= 0 and n2_idx < nsp: Y_g_centers[n2_idx, ~mask] = 1.0
            else: Y_g_centers[0, ~mask] = 1.0

    # --- Update Grid (FVM version) ---
    R_current = max(R_current, 1e-9) # Ensure positive radius for grid
    try:
        r_l_centers, r_l_nodes, volumes_l = grid.liquid_grid_fvm(R_current, Nl)
        r_g_centers, r_g_nodes, volumes_g = grid.gas_grid_fvm(R_current, config.RMAX, Ng)
    except Exception as e:
        print(f"ERROR during grid generation at R={R_current:.2e}: {e}")
        return np.zeros_like(y)

    # --- Assume Pressure (Spatially Uniform) ---
    P = config.P_INIT

    # --- Calculate Gas Density at Cell Centers ---
    rho_g_centers = np.zeros(Ng)
    valid_state = True
    for i in range(Ng):
        # Ensure temperatures are physical before density calc
        T_g_cell = max(200.0, min(T_g_centers[i], 5000.0)) # Clip temperature
        rho_g_centers[i] = gas_props.get_density(T_g_cell, P, Y_g_centers[:,i])
        if np.isnan(rho_g_centers[i]) or rho_g_centers[i] <= 0:
            print(f"Error: Invalid density rho={rho_g_centers[i]:.2e} at gas cell {i}, T={T_g_cell:.1f}. Stopping.")
            valid_state = False; break
    if not valid_state: return np.zeros_like(y)

    # --- Calculate Interface Conditions & Fluxes ---
    if config.LOG_LEVEL >= 2: print(f"    Calculating interface @ t={t:.3e}...", end='\r', flush=True)
    T_s = T_l_centers[Nl-1] # Liquid surface temperature from last liquid cell center

    # Get liquid temperature at center of cell Nl-2 (if Nl>=2)
    T_l_node_last = T_l_centers[Nl-1] # Use Ts if Nl=1
    r_l_node_last_center = r_l_centers[Nl-1] if Nl>=1 else 0.0 # Use center if Nl=1? No, use r=0?
    if Nl >= 2:
        T_l_node_last = T_l_centers[Nl-2]
        r_l_node_last_center = r_l_centers[Nl-2]
    elif Nl == 1:
         T_l_node_last = T_l_centers[0] # Use the only cell center temp
         r_l_node_last_center = r_l_centers[0] # Use the only cell center radius (0)

    try:
        # Calculate fluxes using the FVM interface function
        mdot_double_prime, q_gas_to_surf, q_liq_to_surf, v_g_surf, Y_eq_calc = interface.solve_interface_conditions(
            T_l_node_last=T_l_node_last, # Temp of cell adjacent to surface in liquid
            T_g_node0=T_g_centers[0],    # Temp of cell adjacent to surface in gas
            Y_g_node0=Y_g_centers[:,0], P=P, R=R_current,
            r_l_node_last_center=r_l_node_last_center, # Radius of adjacent liquid cell center
            r_g_node0_center=r_g_centers[0],         # Radius of adjacent gas cell center
            gas_props=gas_props, liquid_props=liquid_props,
            Nl=Nl
        )        
        '''mdot_double_prime, q_gas_to_surf, q_liq_to_surf, v_g_surf, Y_eq_calc = interface.calculate_interface_fluxes(
            T_s=T_s, T_g_node0=T_g_centers[0], Y_g_node0=Y_g_centers[:,0], P=P, R=R_current,
            r_g_node0_center=r_g_centers[0],
            T_l_node_last=T_l_node_last,
            r_l_node_last_center=r_l_node_last_center,
            gas_props=gas_props, liquid_props=liquid_props,
            Nl=Nl # Pass Nl here
        )'''
    except Exception as e:
        print(f"\nERROR during interface calculation at t={t:.4e}, T_s={T_s:.2f}: {e}")
        import traceback; traceback.print_exc()
        return np.zeros_like(y)

    if config.LOG_LEVEL >= 2:
        print(" " * 120, end='\r', flush=True)
        print(f"    DEBUG Interface: mdot'' = {mdot_double_prime:.3e} kg/m2/s, qgas->s={q_gas_to_surf:.2e}, qliq->s={q_liq_to_surf:.2e}, vSurf={v_g_surf:.2e}")


    # --- Calculate RHS ---
    if config.LOG_LEVEL >= 2: print(f"    Calculating liquid RHS @ t={t:.3e}...", end='\r', flush=True)
    try:
        dTl_dt = liquid_phase.calculate_dTl_dt_fvm(
            T_l_centers, r_l_centers, r_l_nodes, volumes_l, liquid_props, Nl,
            q_liq_to_surf=q_liq_to_surf # Pass heat flux FROM liquid TO surface
        )
    except Exception as e:
        print(f"\nERROR during liquid RHS calculation at t={t:.4e}: {e}")
        return np.zeros_like(y)


    if config.LOG_LEVEL >= 2: print(f"    Calculating gas RHS @ t={t:.3e}...", end='\r', flush=True)
    try:
        # Pass correct boundary condition terms
        dTg_dt, dYg_dt = gas_phase.calculate_gas_rhs_fvm(
            T_g_centers, Y_g_centers, rho_g_centers, P, R_current,
            r_g_centers, r_g_nodes, volumes_g, gas_props, Nl, Ng, nsp,
            mdot_double_prime=mdot_double_prime, # Mass flux OUT from surface
            q_gas_to_surf=q_gas_to_surf,     # Heat flux INTO surface FROM gas
            v_g_surf=v_g_surf,             # Gas velocity OUT from surface
            Y_equilibrium=Y_eq_calc        # Equilibrium composition at surface
        )
    except Exception as e:
        print(f"\nERROR during gas RHS calculation at t={t:.4e}: {e}")
        import traceback; traceback.print_exc()
        return np.zeros_like(y)


    # --- Calculate Radius RHS ---
    if config.LOG_LEVEL >= 2: print(f"    Calculating radius RHS @ t={t:.3e}...", end='\r', flush=True)
    rho_l_s = liquid_props.get_prop('density', T_s)
    dR_dt = liquid_phase.calculate_dR_dt(mdot_double_prime, rho_l_s)
    # Safety: Prevent radius growth if mdot is zero/negative
    if mdot_double_prime <= 1e-15: dR_dt = min(0.0, dR_dt)
    # Safety: Limit excessive shrinkage rate relative to current time step maybe?
    # max_shrink_rate = -R_current / (10 * config.DT_MAX) # Limit based on max dt
    # dR_dt = max(dR_dt, max_shrink_rate)


    # Check for NaNs/Infs in derivatives
    fail = False
    if np.any(np.isnan(dTl_dt)) or np.any(np.isinf(dTl_dt)): print(f"\nERROR: NaN/Inf in dTl_dt at t={t:.4e}."); fail=True
    if np.any(np.isnan(dTg_dt)) or np.any(np.isinf(dTg_dt)): print(f"\nERROR: NaN/Inf in dTg_dt at t={t:.4e}."); fail=True
    if np.any(np.isnan(dYg_dt)) or np.any(np.isinf(dYg_dt)): print(f"\nERROR: NaN/Inf in dYg_dt at t={t:.4e}."); fail=True
    if np.isnan(dR_dt) or np.isinf(dR_dt): print(f"\nERROR: NaN/Inf in dR_dt at t={t:.4e}."); fail=True
    if fail:
        print("Returning zero derivatives due to NaN/Inf.")
        return np.zeros_like(y)

    # Assemble dy/dt vector
    dydt = np.concatenate((
        dTl_dt,
        dTg_dt,
        dYg_dt.flatten(),
        np.array([dR_dt])
    ))

    # Update max dT/dt for termination check
    max_dTdt_current = np.max(np.abs(dTg_dt)) if Ng > 0 else 0.0
    if np.isnan(max_dTdt_current): max_dTdt_current = 0.0

    # --- DEBUG STEP: Print dydt summary ---
    if config.LOG_LEVEL >= 2:
        print(" " * 120, end='\r', flush=True)
        max_dTl = np.max(np.abs(dTl_dt)) if Nl > 0 else 0.0
        max_dYg = np.max(np.abs(dYg_dt)) if Ng > 0 else 0.0
        print(f"    DEBUG ODE End t={t:.3e}: dRdt={dR_dt:.3e}, max|dTl/dt|={max_dTl:.3e}, max|dTg/dt|={max_dTdt_current:.3e}, max|dYg/dt|={max_dYg:.3e}")

    return dydt

# --- Main Execution Logic ---
# ... (rest of the run_simulation function remains the same as previous version) ...
# ... (Make sure the call to plotting.plot_results is correct) ...
def run_simulation():
    """Runs the droplet simulation using solve_ivp and FVM."""
    global ode_call_count # Access global counter

    # --- Initialization ---
    print("Initializing simulation (FVM)...")
    start_time_init = time.time()
    try:
        # Initialize property handlers
        liquid_props = properties.LiquidProperties(config.LIQUID_PROP_FILE)
        gas_props = properties.GasProperties(config.MECH_FILE, config.USE_RK_EOS)
        print(f"Using mechanism: {config.MECH_FILE}")
        print(f"Liquid properties: {config.LIQUID_PROP_FILE}")
        print(f"RK EOS Corrections: {'Enabled' if config.USE_RK_EOS else 'Disabled'}")
        print(f"Reaction Type: {config.REACTION_TYPE}")
        print(f"Diffusion Option: {config.DIFFUSION_OPTION}")
        print(f"Advection Scheme: {config.ADVECTION_SCHEME}")
        print(f"Reaction Cutoff: {'Enabled' if config.ENABLE_REACTION_CUTOFF else 'Disabled'} "
              f"(T > {config.REACTION_CALC_MIN_TEMP} K, X_fuel > {config.REACTION_CALC_MIN_FUEL_MOL_FRAC:.1e})")
        print(f"Log Level: {config.LOG_LEVEL}")
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        import traceback; traceback.print_exc(); return

    # Get parameters from config
    nsp = gas_props.nsp
    Nl=config.NL; Ng=config.NG; R0=config.R0

    # Initial State Vectors (represent CELL CENTER values)
    T_l0_centers = np.full(Nl, config.T_L_INIT)
    T_g0_centers = np.full(Ng, config.T_INF_INIT)
    Y_g0_centers = np.zeros((nsp, Ng)) # Initialize Y_g0 array first
    P_current = config.P_INIT

    # Set initial gas composition safely
    try:
        # Use TPX if possible, otherwise TPY
        if hasattr(gas_props.gas, 'TPX'):
             gas_props.gas.TPX = config.T_INF_INIT, P_current, config.X_INF_INIT
        else:
             gas_props.gas.TP = config.T_INF_INIT, P_current
             # Convert X to Y manually if TPX not available
             X_init = np.zeros(nsp)
             for name, x_val in config.X_INF_INIT.items():
                 try: X_init[gas_props.gas.species_index(name)] = x_val
                 except ValueError: pass # Ignore species not in mechanism
             gas_props.gas.X = X_init # Set mole fractions
        initial_Y_array = gas_props.gas.Y # Get mass fractions
        Y_g0_centers[:, :] = initial_Y_array[:, np.newaxis]
        print(f"Initial gas state set: T={gas_props.gas.T:.1f}K, P={gas_props.gas.P:.2e}Pa")
    except Exception as e:
        print(f"CRITICAL ERROR setting initial gas state: {e}")
        return

    # --- Artificial Initial Smoothing (Example for Temperature) ---
    num_smooth_cells = 3 # Smooth over first 3 gas cells
    if Ng >= num_smooth_cells:
        T_s_init = 300 #config.T_L_INIT # Initial surface temp
        for i in range(num_smooth_cells):
            # Simple linear smoothing from T_s_init to T_INF_INIT
            frac = (i + 1.0) / num_smooth_cells
            T_g0_centers[i] = T_s_init + (config.T_INF_INIT - T_s_init) * frac**0.5 # Use sqrt for steeper initial gradient
        # Optionally smooth species as well (e.g., from Y_equilibrium(Ts_init) to Y_ambient)
        # _, Y_eq_init = interface.calculate_fuel_mass_fraction_surf(T_s_init, P_current, gas_props, liquid_props)
        # for i in range(num_smooth_cells):
        #     frac = (i + 1.0) / num_smooth_cells
        #     Y_g0_centers[:, i] = Y_eq_init + (initial_Y_array - Y_eq_init) * frac**0.5
        #     # Renormalize Y after smoothing
        #     Y_g0_centers[:, i] = np.maximum(Y_g0_centers[:, i], 0)
        #     sum_yi = np.sum(Y_g0_centers[:, i])
        #     if sum_yi > 1e-9: Y_g0_centers[:, i] /= sum_yi

    # Combine state into initial vector y0
    # Order: T_l[0]..T_l[Nl-1], T_g[0]..T_g[Ng-1], Y_g[0,0]..Y_g[nsp-1, Ng-1], R(t)
    y0 = np.concatenate(( T_l0_centers, T_g0_centers, Y_g0_centers.flatten(), np.array([R0]) ))
    idx_Tl_end = Nl; idx_Tg_end = Nl + Ng; idx_Yg_end = Nl + Ng + nsp * Ng; idx_R = len(y0) - 1

    # History storage
    saved_times = [0.0]
    initial_state_dict = {
         'T_l': y0[0:idx_Tl_end].copy(),
         'T_g': y0[idx_Tl_end:idx_Tg_end].copy(),
         'Y_g': y0[idx_Tg_end:idx_Yg_end].reshape((nsp, Ng)).copy(),
         'R': y0[idx_R]
    }
    saved_results_list = [initial_state_dict] # List to store states
    max_dTdt_hist = [0.0]

    end_time_init = time.time()
    print(f"Initialization complete ({end_time_init - start_time_init:.2f} s)")
    print(f"Using ODE solver: solve_ivp with BDF method (rtol={config.SOLVER_TOL:.1e}, max_step={config.DT_MAX:.1e}s).")
    abs_tol = config.SOLVER_TOL * config.ATOL_FACTOR * np.abs(y0) # Element-wise absolute tolerance
    abs_tol[abs_tol < 1e-12] = 1e-12 # Set minimum absolute tolerance


    # --- Time Integration Loop ---
    print("-" * 60); print(f"Starting time integration up to t = {config.T_END:.2e} s..."); print("-" * 60)
    current_t = 0.0; step_count_total = 0; ode_call_count = 0
    status = "Running"; start_time_loop = time.time()

    while current_t < config.T_END and status == "Running":
        t_span = (current_t, config.T_END)
        time_before_solve = time.time()

        # Ensure y0 is valid before calling solver
        if np.any(np.isnan(y0)):
            print(f"CRITICAL ERROR: NaN detected in state vector y0 before calling solve_ivp at t={current_t:.4e}. Stopping.")
            status = "NaN Error"; break

        # Call solve_ivp
        try:
            sol = solve_ivp(ode_system, t_span, y0, method='BDF',
                            args=(gas_props, liquid_props),
                            rtol=config.SOLVER_TOL, atol=abs_tol,
                            max_step=config.DT_MAX,
                            first_step=config.DT_INIT,
                            dense_output=True) # Enable dense output for saving
        except Exception as e:
             print(f"\nCRITICAL ERROR occurred during solve_ivp call at t={current_t:.4e} s: {e}")
             import traceback; traceback.print_exc(); status = "Solver Error"; break

        time_after_solve = time.time(); solve_duration = time_after_solve - time_before_solve

        if not sol.success:
            print(f"\nSolver Warning/Error: Solver failed at t={current_t:.4e} s. Status: {sol.status}. Message: {sol.message}")
            # Try to get the last attempted state if available
            if len(sol.y) > 0: y0 = sol.y[:, -1]
            status = "Solver Failed"; break
        if len(sol.t) <= 1 or sol.t[-1] <= current_t + 1e-15: # Check for effective progress
             print(f"\nSolver Warning: No progress made at t={current_t:.4e}s (dt may be too small or stagnant?). Stopping.");
             if len(sol.y) > 0: y0 = sol.y[:, -1]
             status = "Solver Stalled"; break

        # --- Process Results from this Solver Call ---
        last_successful_t = current_t # Store time before update
        current_t = sol.t[-1]
        y0 = sol.y[:, -1] # Update state for the next BDF step
        step_count_total += 1
        num_internal_steps = len(sol.t) -1 if len(sol.t) > 0 else 0 # Internal steps taken by BDF

        # Update absolute tolerance based on the new state
        abs_tol = config.SOLVER_TOL * config.ATOL_FACTOR * np.abs(y0)
        abs_tol[abs_tol < 1e-12] = 1e-12

        # Store max_dTdt history (last value updated globally in ode_system)
        max_dTdt_hist.append(max_dTdt_current)

        # --- Save data periodically using dense output ---
        t_save_next = saved_times[-1] + config.SAVE_INTERVAL_TIME
        save_count_this_step = 0
        while t_save_next <= current_t:
             # Ensure we don't go back in time or save too close to previous point
             if t_save_next > saved_times[-1] + 1e-12:
                 try:
                     y_save = sol.sol(t_save_next) # Interpolate using dense output
                     # Basic check on interpolated state
                     if not np.any(np.isnan(y_save)):
                         saved_times.append(t_save_next)
                         # Unpack state for storage
                         T_l_save = y_save[0:Nl]; T_g_save = y_save[Nl:Nl+Ng]
                         Y_g_save = y_save[Nl+Ng:idx_R].reshape((nsp, Ng)); R_save = y_save[idx_R]
                         saved_results_list.append({
                              'T_l': T_l_save.copy(), 'T_g': T_g_save.copy(),
                              'Y_g': Y_g_save.copy(), 'R': R_save
                         })
                         save_count_this_step += 1
                     else:
                          print(f"Warning: Dense output interpolation returned NaN at t={t_save_next:.4e}. Skipping save.")

                 except Exception as e:
                      print(f"Warning: Error during dense output interpolation/saving at t={t_save_next:.4e}: {e}")

             t_save_next += config.SAVE_INTERVAL_TIME
             # Safety break if loop seems stuck (e.g., interval too small)
             if t_save_next <= last_successful_t and save_count_this_step > 5:
                 print("Warning: Save time loop potentially stuck. Breaking inner loop.")
                 break

        if config.LOG_LEVEL >= 1 and save_count_this_step > 0:
            print(" " * 120, end='\r') # Clear previous line
            print(f" --- Saved {save_count_this_step} states up to t={saved_times[-1]:.4e} s ---")


        # --- Terminal Output ---
        log_now = False
        if config.LOG_LEVEL == 0 and (step_count_total % config.TERMINAL_OUTPUT_INTERVAL_STEP == 0): log_now = True
        elif config.LOG_LEVEL >= 1: log_now = True

        if log_now or status != "Running":
            R_current_val = y0[idx_R]
            Tl_surf_val = y0[Nl - 1]; Tg_surf_val = y0[Nl] # Cell center values
            if config.LOG_LEVEL >= 1: print(" " * 120, end='\r')
            print(f"t={current_t:.4e}s Stp:{step_count_total}(+{num_internal_steps}) ODEs:{ode_call_count} "
                  f"R={R_current_val*1e6:.2f}um Tls={Tl_surf_val:.1f}K Tgs={Tg_surf_val:.1f}K "
                  f"max|dT/dt|={max_dTdt_current:.2e}K/s Dur:{solve_duration:.2f}s")

        # --- Check Termination Conditions ---
        if max_dTdt_current > config.IGNITION_CRITERION_DTDT:
            print(f"\n-------- IGNITION DETECTED at t = {current_t:.6e} s --------")
            status = "Ignited"; break
        if y0[idx_R] < 1e-7: # Check if radius is near zero
             print(f"\n-------- Droplet Evaporated near t = {current_t:.6e} s --------")
             status = "Evaporated"; break
        # Add check for extinction?
        # if current_t > 0.001 and max_dTdt_current < config.EXTINCTION_CRITERION_DTDT:
        #      print(f"\n-------- Extinction Suspected at t = {current_t:.6e} s --------")
        #      status = "Extinguished"; break


    # --- Finalization ---
    end_time_loop = time.time()
    # Save the very last state if simulation didn't fail before first save
    if len(saved_times) > 0 and current_t > saved_times[-1] + 1e-12:
        saved_times.append(current_t)
        y_save = y0 # Use the final state
        T_l_save = y_save[0:Nl]; T_g_save = y_save[Nl:Nl+Ng]; Y_g_save = y_save[Nl+Ng:idx_R].reshape((nsp, Ng)); R_save = y_save[idx_R]
        saved_results_list.append({'T_l': T_l_save.copy(), 'T_g': T_g_save.copy(), 'Y_g': Y_g_save.copy(), 'R': R_save})

    if status == "Running": status = "Ended (Time Limit)"
    print(" " * 120, end='\r') # Clear last status line
    print("-" * 60); print(f"Simulation loop finished at t = {current_t:.6e} s.")
    print(f"Final Status: {status}"); print(f"Total simulation time: {end_time_loop - start_time_loop:.2f} seconds.")
    print(f"Total ODE evaluations: {ode_call_count}"); print("-" * 60)

    # --- Post Processing ---
    if len(saved_results_list) > 1:
        # Make sure plotting function is compatible with FVM results format
        plotting.plot_results(saved_times, saved_results_list, config.OUTPUT_DIR, nsp, Nl, Ng, gas_props)
    else:
        print("Not enough data points saved for plotting.")


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR
    if not os.path.exists(out_dir):
        print(f"Creating output directory: {out_dir}")
        os.makedirs(out_dir)
    run_simulation()
    print("\nProgram finished.")