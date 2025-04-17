# =========================================
#          config.py (修正後)
# =========================================
# config.py
"""Simulation configuration parameters."""

import numpy as np

# --- Initial Conditions ---
T_INF_INIT = 773.0  # K (Initial ambient gas temperature)
P_INIT = 3.0e6     # Pa (Initial pressure)
X_INF_INIT = {'o2': 0.21, 'n2': 0.79} # Initial ambient gas composition (mole fractions)
T_L_INIT = 300.0   # K (Initial liquid temperature)
R0 = 0.05e-3       # m (Initial droplet radius)
FUEL_SPECIES_NAME = 'nc7h16' # Make sure this matches the name in the mechanism file

# --- Grid Parameters ---
NL = 11             # Number of liquid grid points (cell centers)
NG = 41             # Number of gas grid points (cell centers)
R_RATIO = 9.12      # rmax / R0
RMAX = R_RATIO * R0 # Maximum radius of computational domain (m)
GRID_TYPE = 'geometric_series' #Options: 'power_law', 'geometric_series'
XI_GAS_GRID = 2.0 # Parameter for power_law grid (Geometric grid is used now)
# --- Phase Transition ---
R_TRANSITION_RATIO = 0.98
R_TRANSITION_THRESHOLD = R0 * R_TRANSITION_RATIO # m (Radius below which liquid phase calculations stop, e.g., 1 micron)
# --- Time Integration Parameters ---
T_END = 0.006 #5e-7                 # s (Simulation end time)
# DT_INIT is now used only for the VERY first step or as a minimum dt
DT_INIT = 1e-8              # s (Initial/Minimum time step)
# DT_MAX is used to prevent dt from becoming too large
DT_MAX = 1e-4                # s (Maximum allowed time step)
# --- Adaptive Time Stepping Parameters ---
USE_ADAPTIVE_DT = True       # True to enable adaptive time stepping
CFL_NUMBER = 5             # Courant-Friedrichs-Lewy number (safety factor, typically < 1)
DT_MAX_INCREASE_FACTOR = 1.5 # Maximum factor by which dt can increase per step (e.g., 1.2 = 20% increase)
DT_MIN_VALUE = 1e-15         # Absolute minimum dt allowed
TIME_STEPPING_MODE = 'BDF'   # Currently uses solve_ivp with BDF

# --- Model Options ---
USE_RK_EOS = False #True             # True to use Redlich-Kwong EOS for density
# --- !!! REACTION_TYPE: Choose 'detailed', 'overall', or 'none' !!! ---
REACTION_TYPE = 'detailed'        # Options: 'detailed', 'overall', 'none'
# --- !!! DIFFUSION_OPTION: Choose 'constant', 'Le=1', or 'mixture_averaged' !!! ---
DIFFUSION_OPTION = 'mixture_averaged'     # Options: 'constant', 'Le=1', 'mixture_averaged'
# --- !!! ADVECTION_SCHEME: Choose 'upwind' or 'central' !!! ---
ADVECTION_SCHEME = 'upwind'   # Options: 'upwind', 'central'

# --- Overall Reaction Parameters (if REACTION_TYPE = 'overall') ---
OVERALL_B_CM = 4.4e16     # cm^3 / (mol * s)
OVERALL_E_KJ = 209.2      # kJ / mol
OVERALL_E_SI = OVERALL_E_KJ * 1000.0 # J / mol
OVERALL_B_SI = OVERALL_B_CM * 1e-6 # m^3 / (mol * s)
OVERALL_FUEL = FUEL_SPECIES_NAME
OVERALL_OXIDIZER = 'o2'

# --- Diffusion Parameter (if DIFFUSION_OPTION = 'constant') ---
DIFFUSION_CONSTANT_VALUE = 1e-5 # m^2/s

# --- Files ---
LIQUID_PROP_FILE = 'n_heptane_liquid_properties.csv'
MECH_FILE = 'mech_LLNL_reduce.yaml'

# --- Termination Criteria ---
IGNITION_CRITERION_DTDT = 1.0e7  # K/s (Max gas temperature rise rate)
EXTINCTION_CRITERION_DTDT = 1.0    # K/s (Max gas temperature rise rate threshold for extinction)

# --- Logging Options ---
LOG_LEVEL = 2  # 0: Basic, 1: Info (ODE calls), 2: Debug (Inside ODE, dydt values)
TERMINAL_OUTPUT_INTERVAL_STEP = 1 # Steps (Interval for printing status for LOG_LEVEL 0)

# --- Reaction Calculation Cutoff ---
ENABLE_REACTION_CUTOFF = True     # True to enable cutoff, False to calculate everywhere
REACTION_CALC_MIN_TEMP = 600.0    # K (Minimum temperature to calculate reactions)
REACTION_CALC_MIN_FUEL_MOL_FRAC = 1e-8 # mol/mol (Minimum fuel mole fraction)

# --- Output Options ---
OUTPUT_DIR = 'results_fvm_rk' # Changed output dir name
SAVE_INTERVAL_TIME = 0.0001   # s (Interval to save full data)
PLOT_INTERVAL_TIME = 0.0001   # s (Interval for final plot points)

# --- Physical Constants ---
R_UNIVERSAL = 8.31446261815324 # J/(mol·K)

# --- Numerical Parameters ---
SOLVER_TOL = 1e-4 # Relative tolerance for solve_ivp 1e-6
ATOL_FACTOR = 1e-6 # Absolute tolerance factor (atol = rtol * ATOL_FACTOR * typical_value) 1e-8
MAX_ITER_RK = 10   # Max iterations for RK EOS solver

# --- Redlich-Kwong Parameters ---
# Critical Temp (K), Critical Press (Pa), Acentric factor (-)
# Values from NIST, Dortmund Data Bank, Reid et al. (verify consistency)
RK_PARAMS = {
    'n2':     {'Tc': 126.19, 'Pc': 3.3958e6, 'omega': 0.0372},
    'o2':     {'Tc': 154.58, 'Pc': 5.043e6,  'omega': 0.0222},
    'nc7h16': {'Tc': 540.2,  'Pc': 2.74e6,   'omega': 0.351}, # n-Heptane
    'co2':    {'Tc': 304.13, 'Pc': 7.3773e6, 'omega': 0.2239},
    'h2o':    {'Tc': 647.10, 'Pc': 22.064e6, 'omega': 0.3449}
    # Add others if significant concentrations are expected AND reliable data exists
}
RK_MIXING_RULE = 'van_der_waals' # Simple VdW mixing rules
RK_SOLVER_TOL = 1e-7 # Tolerance for solving cubic EOS