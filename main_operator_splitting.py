# =========================================
#     main_operator_split.py (新規作成)
# =========================================
# main_operator_split.py
"""
Main script for droplet evap/combustion simulation using Operator Splitting (Step 1).
Uses a custom fixed time step loop with Forward Euler for explicit terms.
Reactions are currently OFF.
"""

import numpy as np
import cantera as ct
# Removed: from scipy.integrate import solve_ivp
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Import simulation modules
import config
import grid
import properties
import interface
import reactions # Still needed for structure, but wdot will be zero
import numerics
import liquid_phase # FVM version
import gas_phase    # FVM version
import plotting

# Global variables (consider class structure later)
max_dTdt_current = 0.0
step_count_total = 0

def run_simulation_split():
    """Runs the droplet simulation using a custom time loop with adaptive dt."""
    global max_dTdt_current, step_count_total

    # --- Initialization ---
    print("Initializing simulation (Operator Splitting - Adaptive dt - Single Interface Solve)...")
    start_time_init = time.time()
    try:
        liquid_props = properties.LiquidProperties(config.LIQUID_PROP_FILE)
        gas_props = properties.GasProperties(config.MECH_FILE, config.USE_RK_EOS)
        print(f"Mechanism: {config.MECH_FILE}, Liquid props: {config.LIQUID_PROP_FILE}")
        print(f"RK EOS: {config.USE_RK_EOS}, Diffusion: {config.DIFFUSION_OPTION}, Advection: {config.ADVECTION_SCHEME}")
        print(f"Reaction Type: {config.REACTION_TYPE} (Forced OFF in Step 1)")
        print(f"Log Level: {config.LOG_LEVEL}")
    except Exception as e: print(f"CRITICAL ERROR during initialization: {e}"); return

    nsp = gas_props.nsp
    Nl = config.NL; Ng = config.NG; R0 = config.R0

    # --- Initial State (Cell Centers) ---
    T_l = np.full(Nl, config.T_L_INIT); T_g = np.full(Ng, config.T_INF_INIT)
    Y_g = np.zeros((nsp, Ng)); P = config.P_INIT; R = R0
    try:
        if hasattr(gas_props.gas, 'TPX'): gas_props.gas.TPX = config.T_INF_INIT, P, config.X_INF_INIT
        else: gas_props.gas.TP = config.T_INF_INIT, P; gas_props.gas.X = config.X_INF_INIT
        initial_Y_array = gas_props.gas.Y
        initial_Y_array = np.maximum(initial_Y_array, 0); initial_Y_array /= np.sum(initial_Y_array)
        Y_g[:, :] = initial_Y_array[:, np.newaxis]
        print(f"Initial gas state set: T={gas_props.gas.T:.1f}K, P={gas_props.gas.P:.2e}Pa")
    except Exception as e: print(f"CRITICAL ERROR setting initial gas state: {e}"); return

    # --- History Storage ---
    current_t = 0.0; dt = config.DT_INIT # Use DT_INIT for first step estimate
    saved_times = [current_t]
    initial_state_dict = {'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R}
    saved_results_list = [initial_state_dict]; max_dTdt_hist = [0.0]

    end_time_init = time.time()
    print(f"Initialization complete ({end_time_init - start_time_init:.2f} s)")
    print(f"Using Adaptive time step starting near dt = {dt:.2e} s (CFL={config.CFL_NUMBER})")
    print("-" * 60); print(f"Starting time integration up to t = {config.T_END:.2e} s..."); print("-" * 60)
    status = "Running"; start_time_loop = time.time()
    step_count_total = 0

    is_gas_phase_only = False # Flag for simulation phase
    # --- CSV File Setup ---
    csv_filename = os.path.join(config.OUTPUT_DIR, 'time_history_live.csv')
    # Write header if file doesn't exist
    if not os.path.exists(csv_filename):
        header_df = pd.DataFrame(columns=['Time (s)', 'Radius (m)', 'T_liquid_surf_cell (K)', 'T_gas_surf_cell (K)', 'T_solved_interface (K)', 'Mdot (kg/m2/s)'])
        header_df.to_csv(csv_filename, index=False)
        print(f"Created CSV log file: {csv_filename}")
    # --- End CSV Setup ---

    # --- Custom Time Integration Loop ---
    while current_t < config.T_END and status == "Running":
        step_start_time = time.time()

        # --- 0. Store previous state ---
        T_l_old = T_l.copy(); T_g_old = T_g.copy(); Y_g_old = Y_g.copy(); R_old = R

        # --- 1. Update Grid (based on R_old) ---
        R_safe = max(R_old, config.R_TRANSITION_THRESHOLD * 0.5) if is_gas_phase_only else max(R_old, 1e-9) # Keep small R for grid if gas only
        try:
            # For gas only phase, grid doesn't change much if R is fixed
            r_l_centers, r_l_nodes, volumes_l = grid.liquid_grid_fvm(R_safe, Nl if not is_gas_phase_only else 0) # Use Nl=0 if gas only?
            r_g_centers, r_g_nodes, volumes_g = grid.gas_grid_fvm(R_safe, config.RMAX, Ng)
            A_g_faces = grid.face_areas(r_g_nodes)
        except Exception as e: print(f"ERROR grid gen R={R_safe:.2e}: {e}"); status="Grid Error"; break
        
        # --- 2. Calculate Interface Conditions ONCE (using state at t = current_t) ---
        if not is_gas_phase_only:
            if config.LOG_LEVEL >= 2: print(f"  Calculating interface @ t={current_t:.3e}...", end='\r', flush=True)
            # Determine adjacent liquid cell temp/radius
            if Nl >= 2: T_l_last = T_l_old[Nl-2]; r_l_last_c = r_l_centers[Nl-2]
            elif Nl == 1: T_l_last = T_l_old[0]; r_l_last_c = r_l_centers[0]
            else: T_l_last = config.T_L_INIT; r_l_last_c = 0.0 # Fallback if Nl=0
            # Solve for interface state
            try:
                mdot_double_prime, q_gas_out_surf, q_liq_out_surf, v_g_surf, Y_eq_calc, T_s_solved = interface.solve_interface_conditions(
                    T_l_node_last=T_l_last, T_g_node0=T_g_old[0], Y_g_node0=Y_g_old[:,0], P=P, R=R_old,
                    r_l_node_last_center=r_l_last_c, r_g_node0_center=r_g_centers[0],
                    gas_props=gas_props, liquid_props=liquid_props, Nl=Nl
                )
            except Exception as e: print(f"\nERROR interface solve: {e}"); status = "Interface Error"; break
            if config.LOG_LEVEL >= 2:
                print(" " * 120, end='\r', flush=True)
                print(f"    DEBUG Interface (Solved): Ts={T_s_solved:.2f}K mdot''={mdot_double_prime:.3e}, qgas->s={q_gas_out_surf:.2e}, qliq->s={q_liq_out_surf:.2e}, vSurf={v_g_surf:.2e}")
        else:
            # Gas phase only: Set interface fluxes to zero
            mdot_double_prime = 0.0; q_gas_out_surf = 0.0; q_liq_out_surf = 0.0
            v_g_surf = 0.0; Y_eq_calc = Y_g_old[:, 0].copy(); T_s_solved = T_g_old[0]

        # --- 3. Calculate Cell Center Properties (using state at t) ---
        rho_g = np.zeros(Ng); rho_l = np.zeros(Nl)
        cp_g = np.zeros(Ng); cp_l = np.zeros(Nl)
        lambda_g = np.zeros(Ng); lambda_l = np.zeros(Nl)
        Dk_g = np.zeros((nsp, Ng)); h_k_g = np.zeros((nsp, Ng))
        valid_state = True
        # Gas props
        for i in range(Ng):
             T_g_cell = max(200.0, min(T_g_old[i], 5000.0))
             if not gas_props.set_state(T_g_cell, P, Y_g_old[:,i]): valid_state = False; break
             rho_g[i] = gas_props.get_density(T_g_cell, P, Y_g_old[:,i])
             cp_g[i] = gas_props.gas.cp_mass
             lambda_g[i] = gas_props.gas.thermal_conductivity
             Dk_g[:, i] = gas_props.get_diffusion_coeffs(T_g_cell, P, Y_g_old[:,i])
             h_k_g[:, i] = gas_props.get_partial_enthalpies_mass(T_g_cell, P, Y_g_old[:, i])
             if np.isnan(rho_g[i]) or rho_g[i] <= 0: valid_state = False; break
        # Liquid props
        for j in range(Nl):
             T_l_cell = max(200.0, min(T_l_old[j], 2000.0))
             props_l = liquid_props.get_properties(T_l_cell)
             rho_l[j] = props_l.get('density', np.nan); cp_l[j] = props_l.get('specific_heat', np.nan); lambda_l[j] = props_l.get('thermal_conductivity', np.nan)
             if np.isnan(rho_l[j]) or np.isnan(cp_l[j]) or np.isnan(lambda_l[j]): valid_state=False; break
        if not valid_state: print("Error getting props for dt calc."); status="Prop Error"; break

        # --- 4. Calculate Face Velocities (using mdot'' from step 2) ---
        #A_g_faces = grid.face_areas(r_g_nodes)
        u_g_faces = np.zeros(Ng + 1)
        mass_flux_faces = np.zeros(Ng + 1)
        mass_flux_faces[0] = mdot_double_prime * A_g_faces[0]
        u_g_faces[0] = v_g_surf # Use solved surface velocity
        Mdot = mass_flux_faces[0]
        for i in range(1, Ng):
             mass_flux_faces[i] = Mdot # Quasi-steady mass flow
             rho_face_i = numerics.arithmetic_mean(rho_g[i-1], rho_g[i])
             if A_g_faces[i] > 1e-15 and rho_face_i > 1e-6: u_g_faces[i] = Mdot / (rho_face_i * A_g_faces[i])
             else: u_g_faces[i] = 0.0
        mass_flux_faces[Ng] = 0.0; u_g_faces[Ng] = 0.0

        # --- 5. Calculate Adaptive Time Step 'dt' ---
        dt = numerics.calculate_adaptive_dt(
             u_g_faces, lambda_g, rho_g, cp_g, Dk_g, # Gas props/vel
             lambda_l, rho_l, cp_l, # Liquid props
             r_g_nodes, r_l_nodes, # Grids
             dt, # Pass previous dt
             nsp # Number of species
             )
        dt = min(dt, config.T_END - current_t + 1e-15) # Prevent overshooting T_END
        if dt < config.DT_MIN_VALUE: status = "dt too small"; break

        # --- 6. Explicit Advection Step (Forward Euler for Phi*rho*V) ---
        #    Calculate d(Phi)/dt due to advection only
        if config.LOG_LEVEL >= 2: print(f"  Calculating advection @ t={current_t:.3e}...", end='\r', flush=True)
        dTg_dt_adv, dYg_dt_adv = gas_phase.calculate_gas_advection_rhs(
             T_g_old, Y_g_old, rho_g, cp_g, h_k_g, u_g_faces, # Use props at t
             r_g_nodes, volumes_g, Ng, nsp)
        # Liquid phase has no advection in this model
        dTl_dt_adv = np.zeros(Nl)

        # Intermediate state after advection step (Euler step)
        T_l_star = T_l_old + dt * dTl_dt_adv # T_l_star = T_l_old
        T_g_star = T_g_old + dt * dTg_dt_adv
        Y_g_star = Y_g_old + dt * dYg_dt_adv
        # Normalize Y_g_star
        Y_g_star = np.maximum(Y_g_star, 0.0)
        sum_Yg_star = np.sum(Y_g_star, axis=0)
        mask_star = sum_Yg_star > 1e-9
        Y_g_star[:, mask_star] /= sum_Yg_star[mask_star]
        if np.any(~mask_star): Y_g_star[:, ~mask_star] = 0.0; Y_g_star[gas_props.n2_idx if gas_props.n2_idx>=0 else 0, ~mask_star] = 1.0

        # --- 7. Implicit Diffusion Step (Backward Euler using T_star, Y_star as RHS) ---
        if config.LOG_LEVEL >= 2: print(f"  Calculating diffusion @ t={current_t:.3e}...", end='\r', flush=True)
        # --- 7a. Liquid Temperature Diffusion ---
        if Nl > 0:
             if not is_gas_phase_only:
                a_l, b_l, c_l, d_l = liquid_phase.build_diffusion_matrix_liquid(
                    T_l_star, r_l_centers, r_l_nodes, volumes_l, liquid_props, Nl, dt,
                    q_liq_out_surf=q_liq_out_surf # Use flux calculated at step 2
                )
                if a_l is None: status = "Matrix Build Error"; break
                T_l_new = numerics.solve_tridiagonal(a_l, b_l, c_l, d_l)
                if T_l_new is None: status = "Matrix Solve Error"; break
                T_l = T_l_new # Update liquid temperature
             else:
                 T_l = np.zeros([Nl])
        else: T_l = np.array([])

        # --- 7b. Gas Temperature Diffusion ---
        if Ng > 0:
             # Need properties (rho, cp, lambda) potentially at T_star state?
             # For simplicity of Backward Euler, use props from T_old (or iterate)
             # Let's use props from T_old for now.
             a_gT, b_gT, c_gT, d_gT = gas_phase.build_diffusion_matrix_gas_T(
                 T_g_star, rho_g, cp_g, lambda_g, # Use T_star in RHS, props from T_old
                 r_g_centers, r_g_nodes, volumes_g, Ng, dt,
                 q_gas_out_surf=q_gas_out_surf, # Use flux calculated at step 2
                 A_g_faces=A_g_faces)
             if a_gT is None: status = "Matrix Build Error"; break
             T_g_new = numerics.solve_tridiagonal(a_gT, b_gT, c_gT, d_gT)
             if T_g_new is None: status = "Matrix Solve Error"; break
             T_g = T_g_new # Update gas temperature
        else: T_g = np.array([])

        # --- 7c. Gas Species Diffusion (solve for each species) ---
        fuel_idx = gas_props.fuel_idx
        if Ng > 0:
             Y_g_new = np.zeros_like(Y_g_star)
             # Need species flux at boundary for each species
             SpeciesFlux_face0 = np.zeros(nsp)
             SpeciesFlux_face0[fuel_idx] = mdot_double_prime * A_g_faces[0] # Fuel flux out

             for k in range(nsp):
                 # Need Dk and rho based on T_old or T_star? Use T_old for now.
                 a_gy, b_gy, c_gy, d_gy = gas_phase.build_diffusion_matrix_gas_Y(
                     k, Y_g_star, rho_g, Dk_g, # Use Y_star in RHS, props from T_old
                     r_g_centers, r_g_nodes, volumes_g, Ng, nsp, dt,
                     SpeciesFlux_face0_k=SpeciesFlux_face0[k], # Use boundary flux from step 2
                     A_g_faces=A_g_faces)
                 if a_gy is None: status = "Matrix Build Error"; break
                 Yk_new = numerics.solve_tridiagonal(a_gy, b_gy, c_gy, d_gy)
                 if Yk_new is None: status = "Matrix Solve Error"; break
                 Y_g_new[k, :] = Yk_new
             if status != "Running": break # Exit if solve failed for any species
             Y_g = Y_g_new # Update species array
        else: Y_g = np.zeros((nsp, 0))

        # --- 8. Explicit Radius Update (Forward Euler) ---
        # Use mdot'' calculated at beginning of step based on T_old state
        if not is_gas_phase_only:
            rho_l_s = liquid_props.get_prop('density', T_s_solved) if Nl > 0 else 1000.0
            dR_dt = liquid_phase.calculate_dR_dt(mdot_double_prime, rho_l_s)
            if mdot_double_prime <= 1e-15: dR_dt = min(0.0, dR_dt)
        else:
            dR_dt = 0.0 # Radius is fixed
        R = R_old + dt * dR_dt # Update Radius
        
        R = max(R, 1e-9) # Keep radius positive
        # ---  Check for Phase Transition ---
        if R < config.R_TRANSITION_THRESHOLD and not is_gas_phase_only:
            print(f"\n--- Transitioning to Gas Phase Only at t={current_t:.4e} s (R={R:.2e} m) ---")
            is_gas_phase_only = True
            # Keep R constant after transition? Or set to 0 for grid? Let's keep it small.
            R = config.R_TRANSITION_THRESHOLD

        # --- 9. Update Time and Step Count ---
        current_t += dt
        step_count_total += 1

        # --- 10. Data Saving & Logging ---
        # ... (Save data based on config.SAVE_INTERVAL_TIME) ...
        # ... (Log output based on config.LOG_LEVEL and interval) ...
        save_now = False
        if current_t >= saved_times[-1] + config.SAVE_INTERVAL_TIME - 1e-12:
             if current_t > saved_times[-1] : # Avoid saving at exact same time
                 save_now = True
                 # Check for NaN before saving
                 if not (np.any(np.isnan(T_l)) or np.any(np.isnan(T_g)) or np.any(np.isnan(Y_g)) or np.isnan(R)):
                     saved_times.append(current_t)
                     saved_results_list.append({'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R})

                 else: print(f"Warning: NaN detected in state at t={current_t:.4e}. Not saving this step.")

                # --- ★ Append current state to CSV ---
                 try:
                     Tls_now = T_l[Nl-1] if Nl > 0 and not is_gas_phase_only else np.nan # Save NaN if gas only
                     Tgs_now = T_g[0] if Ng > 0 else np.nan
                     R_now = R
                     mdot_now = mdot_double_prime if not is_gas_phase_only else 0.0
                     Ts_log = T_s_solved if not is_gas_phase_only else Tgs_now # Log solved Ts or gas temp
                     current_data = pd.DataFrame({
                         'Time (s)': [current_t], 'Radius (m)': [R_now],
                         'T_liquid_surf_cell (K)': [np.nan_to_num(Tls_now)],
                         'T_gas_surf_cell (K)': [np.nan_to_num(Tgs_now)],
                         'T_solved_interface (K)': [np.nan_to_num(Ts_log)],
                         'Mdot (kg/m2/s)': [mdot_now]
                     })
                     write_header = not os.path.exists(csv_filename)
                     current_data.to_csv(csv_filename, mode='a', header=write_header, index=False, float_format='%.6e')
                 except Exception as e_csv_live: print(f"Warning: Error appending to live CSV: {e_csv_live}")

        log_now = (config.LOG_LEVEL > 0 and step_count_total % 1 == 0) or \
                  (config.LOG_LEVEL == 0 and step_count_total % config.TERMINAL_OUTPUT_INTERVAL_STEP == 0)

        if log_now or save_now or status != "Running":
             # Use calculated derivatives from THIS step for logging max|dT/dt|
             max_dTdt_current = np.max(np.abs(dTg_dt_adv)) if Ng > 0 else 0.0
             Tl_s_cell = T_l[Nl-1] if Nl > 0 else 0.0
             Tg_s_cell = T_g[0] if Ng > 0 else 0.0
             step_stop_time = time.time(); step_duration = step_stop_time - step_start_time
             # Ensure we have space to clear previous log lines
             print(f"t={current_t:.4e}s dt={dt:.2e} Stp:{step_count_total} R={R*1e6:.2f}um "
                   f"Tls={Tl_s_cell:.1f}K Tgs={Tg_s_cell:.1f}K "
                   f"max|dTg/dt|={max_dTdt_current:.2e}K/s mdot={mdot_double_prime:.2e} "
                   f"Dur:{step_duration:.3f}s" + " "*10) # Add spaces to clear line

             if save_now and config.LOG_LEVEL >= 1: print(f" --- Saved state ---")

        # --- 11. Check Termination Conditions ---
        #if R < 1e-8: status = "Evaporated"; break
        if step_count_total > 2000000: status = "Max Steps"; break # Increase safety break further?

    # --- Finalization ---
    end_time_loop = time.time()
    # Ensure the very last state is saved if loop finished/broke
    if current_t > saved_times[-1] + 1e-12:
        saved_times.append(current_t)
        saved_results_list.append({'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R})

    if status == "Running": status = "Ended (Time Limit)"
    print("-" * 60); print(f"Simulation loop finished at t = {current_t:.6e} s.")
    print(f"Final Status: {status}"); print(f"Total simulation time: {end_time_loop - start_time_loop:.2f} seconds.")
    print(f"Total steps: {step_count_total}"); print("-" * 60)

    # --- Post Processing ---
    if len(saved_results_list) > 1:
        plotting.plot_results(saved_times, saved_results_list, config.OUTPUT_DIR, nsp, Nl, Ng, gas_props)
    else:
        print("Not enough data points saved for plotting.")


if __name__ == "__main__":
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    try:
         run_simulation_split() # Run the new function
    except Exception as main_e:
         print("\n" + "="*20 + " CRITICAL ERROR IN MAIN SCRIPT " + "="*20)
         import traceback
         traceback.print_exc()
         print("="*60)
    finally:
        print("\nProgram finished.")