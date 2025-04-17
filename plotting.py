# =========================================
#         plotting.py (修正後)
# =========================================
# plotting.py
"""Functions for plotting simulation results (works with FVM cell center data)."""

import matplotlib.pyplot as plt
import numpy as np
import os
import config
import grid # To regenerate grids for plotting
from properties import GasProperties # For type hinting

def plot_results(times, results_list, output_dir, nsp, Nl, Ng, gas_props: GasProperties):
    """
    Generates and saves plots from a list of state dictionaries (cell center values).
    Accepts gas_props object to correctly identify fuel species index.
    """
    print("Generating final plots...")
    # Ensure output dir exists, handle potential race condition
    os.makedirs(output_dir, exist_ok=True)

    times_array = np.array(times)
    num_times = len(results_list)
    if num_times < 1: # Need at least one point (initial state)
        print("Not enough time points to plot.")
        return

    # --- Extract Data ---
    Tl_surf_hist = np.zeros(num_times)
    Tg_surf_hist = np.zeros(num_times)
    R_hist = np.zeros(num_times)
    plot_data_valid = True

    for i, state_dict in enumerate(results_list):
        # Check if keys exist and arrays have expected length
        if 'T_l' in state_dict and len(state_dict['T_l']) == Nl and Nl > 0:
            Tl_surf_hist[i] = state_dict['T_l'][-1] # Last liquid cell center
        else: Tl_surf_hist[i] = np.nan

        if 'T_g' in state_dict and len(state_dict['T_g']) == Ng and Ng > 0:
            Tg_surf_hist[i] = state_dict['T_g'][0]  # First gas cell center
        else: Tg_surf_hist[i] = np.nan

        if 'R' in state_dict: R_hist[i] = state_dict['R']
        else: R_hist[i] = np.nan

    # Check if any essential data extraction failed
    if np.isnan(R_hist).any():
         print("Warning: NaN values encountered in Radius history. Plotting might fail.")
         plot_data_valid = False
    if np.isnan(Tl_surf_hist).any() and Nl > 0:
         print("Warning: NaN values encountered in Tl_surf history.")
         # plot_data_valid = False # Allow plotting even if some temp points are NaN
    if np.isnan(Tg_surf_hist).any() and Ng > 0:
         print("Warning: NaN values encountered in Tg_surf history.")
         # plot_data_valid = False

    # Replace NaNs with placeholder values if needed for plotting, but warn
    Tl_surf_hist = np.nan_to_num(Tl_surf_hist, nan=config.T_L_INIT)
    Tg_surf_hist = np.nan_to_num(Tg_surf_hist, nan=config.T_INF_INIT)
    R_hist_plot = np.nan_to_num(R_hist, nan=0.0) # Use R_hist_plot for plotting


    # --- Plot 1: Surface Temperatures vs Time ---
    if Nl > 0 and Ng > 0: # Only plot if both phases exist
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(times_array, Tl_surf_hist, 'b.-', markersize=4, label=f'Liquid Surf Cell (j={Nl-1}) Temp')
            plt.plot(times_array, Tg_surf_hist, 'r.-', markersize=4, label=f'Gas Surf Cell (i=0) Temp')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperatures at Surface Cells vs Time')
            plt.legend()
            plt.grid(True)
            min_temp = min(np.min(Tl_surf_hist[~np.isnan(Tl_surf_hist)]) if not np.all(np.isnan(Tl_surf_hist)) else config.T_L_INIT,
                           np.min(Tg_surf_hist[~np.isnan(Tg_surf_hist)]) if not np.all(np.isnan(Tg_surf_hist)) else config.T_INF_INIT)
            max_temp = max(np.max(Tl_surf_hist[~np.isnan(Tl_surf_hist)]) if not np.all(np.isnan(Tl_surf_hist)) else config.T_L_INIT,
                           np.max(Tg_surf_hist[~np.isnan(Tg_surf_hist)]) if not np.all(np.isnan(Tg_surf_hist)) else config.T_INF_INIT)
            plt.ylim(bottom=max(0, min_temp - 50), top=max_temp + 100)
            plt.savefig(os.path.join(output_dir, 'surface_cell_temperatures.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating surface temperature plot: {e}")
            plt.close()


    # --- Plot 2: Droplet Radius vs Time ---
    if plot_data_valid:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(times_array, R_hist_plot * 1e6, 'k.-', markersize=4) # Radius in micrometers
            plt.xlabel('Time (s)')
            plt.ylabel('Droplet Radius (μm)')
            plt.title('Droplet Radius vs Time')
            plt.grid(True)
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(output_dir, 'droplet_radius.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating droplet radius plot: {e}")
            plt.close()

    # --- Plot 3: Temperature Profiles at Different Times ---
    if plot_data_valid:
        try:
            plt.figure(figsize=(12, 7))
            num_plots = min(num_times, 6)
            plot_indices = []
            # Select indices more intelligently - e.g., log spacing in time?
            # Or ensure last point is included. Simple linspace for now.
            if num_times > 0: plot_indices = np.linspace(0, num_times - 1, num_plots, dtype=int)

            plotted_count = 0
            for i in plot_indices:
                t_plot = times_array[i]
                state_dict = results_list[i]
                T_l_plot = state_dict.get('T_l')
                T_g_plot = state_dict.get('T_g')
                R_plot = state_dict.get('R')

                if T_l_plot is None or T_g_plot is None or R_plot is None or np.isnan(R_plot) \
                   or len(T_l_plot) != Nl or len(T_g_plot) != Ng:
                    print(f"Warning: Missing/invalid data at index {i} (t={t_plot:.3e}). Skipping profile plot.")
                    continue

                R_plot = max(R_plot, 1e-9)

                # Regenerate grids (cell centers) for this radius
                r_l_centers_plot, _, _ = grid.liquid_grid_fvm(R_plot, Nl)
                r_g_centers_plot, _, _ = grid.gas_grid_fvm(R_plot, config.RMAX, Ng)

                # Ensure r_l starts at 0 if Nl>0
                r_l_centers_plot_fixed = r_l_centers_plot.copy() if Nl>0 else np.array([])
                if Nl > 0: r_l_centers_plot_fixed[0] = 0.0

                # Combine for plotting (mirror liquid phase cell centers)
                r_combined = np.concatenate((r_l_centers_plot_fixed, r_g_centers_plot))
                T_combined = np.concatenate((T_l_plot, T_g_plot))
                plt.plot(r_combined / config.R0, T_combined, '.-', label=f't = {t_plot:.3f} s', markersize=4)
                plotted_count += 1

            plt.xlabel('Radius/InitialRadius (-)')
            plt.ylabel('Temperature (K) at Cell Centers')
            plt.title(f'Temperature Profiles (rmax={config.RMAX*1e3:.1f} mm)')
            if plotted_count > 0 : plt.legend()
            plt.grid(True)
            plt.xlim(0, np.ceil(config.R_RATIO))
            plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
            plt.savefig(os.path.join(output_dir, 'temperature_profiles.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating temperature profile plot: {e}")
            plt.close()

    # --- Plot 4: Fuel Mass Fraction Profile ---
    if plot_data_valid:
        try:
            plt.figure(figsize=(12, 7))
            fuel_idx = -1
            if gas_props and hasattr(gas_props, 'fuel_idx'): fuel_idx = gas_props.fuel_idx
            else: print("Warning: gas_props object invalid for fuel fraction plot.")

            plotted_count = 0
            if fuel_idx != -1 and num_times > 0:
                plot_indices = np.linspace(0, num_times - 1, num_plots, dtype=int)
                for i in plot_indices:
                    t_plot = times_array[i]
                    state_dict = results_list[i]
                    Y_g_plot = state_dict.get('Y_g')
                    R_plot = state_dict.get('R')

                    if Y_g_plot is None or R_plot is None or np.isnan(R_plot) \
                       or Y_g_plot.shape != (nsp, Ng):
                        print(f"Warning: Missing/invalid Y_g or R data at index {i}. Skipping fuel profile.")
                        continue

                    R_plot = max(R_plot, 1e-9)
                    r_g_centers_plot, _, _ = grid.gas_grid_fvm(R_plot, config.RMAX, Ng)

                    plt.plot(r_g_centers_plot / config.R0, Y_g_plot[fuel_idx, :], '.-', label=f't = {t_plot:.3f} s', markersize=4)
                    plotted_count += 1

            if plotted_count > 0:
                plt.xlabel('Radius/InitialRadius (-)')
                plt.ylabel(f'Fuel ({config.FUEL_SPECIES_NAME}) Mass Fraction at Cell Centers')
                plt.title('Fuel Mass Fraction Profiles')
                plt.legend()
                plt.grid(True)
                plt.xlim(0, np.ceil(config.R_RATIO))
                plt.ylim(bottom=-0.05, top=1.05) # Allow slightly below 0 for visibility
                plt.savefig(os.path.join(output_dir, 'fuel_fraction_profiles.png'))
            elif fuel_idx == -1:
                print("Skipping fuel fraction plot: fuel index not found.")
            else:
                print("No valid data points to plot fuel fraction.")

            plt.close()
        except Exception as e:
            print(f"Error generating fuel fraction plot: {e}")
            plt.close()


    print(f"Plots saved in '{output_dir}' directory.")