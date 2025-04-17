# =========================================
#        properties.py (修正後)
# =========================================
# properties.py
"""
Handles liquid and gas phase properties.
Includes reading liquid data from CSV and interfacing with Cantera for gas,
with implemented Redlich-Kwong corrections for density.
Implements diffusion coefficient options.
"""
import numpy as np
import pandas as pd
import cantera as ct
from scipy.interpolate import interp1d
from scipy.optimize import newton # Using Newton solver for cubic EOS
import warnings
import config

# --- Suppress RuntimeWarning from numpy roots for complex results ---
#warnings.filterwarnings('ignore', category=np.ComplexWarning)

class LiquidProperties:
    """Reads and interpolates liquid properties from a CSV file."""
    def __init__(self, filename=config.LIQUID_PROP_FILE):
        try:
            self.df = pd.read_csv(filename)
            self.df.sort_values(by='temperature', inplace=True)
            if self.df.isnull().values.any():
                print(f"Warning: NaN values found in {filename}. Check the file.")
                self.df.fillna(method='ffill', inplace=True) # Simple fill strategy
                self.df.fillna(method='bfill', inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Liquid property file '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading liquid property file '{filename}': {e}")

        self._create_interpolators()
        self.T_min = self.df['temperature'].min()
        self.T_max = self.df['temperature'].max()

    def _create_interpolators(self):
        self.interpolators = {}
        for col in self.df.columns:
            if col != 'temperature':
                self.interpolators[col] = interp1d(
                    self.df['temperature'], self.df[col],
                    kind='linear', # Linear might be more stable than cubic
                    fill_value="extrapolate",
                    bounds_error=False
                )

    def get_properties(self, T):
        """Returns a dictionary of liquid properties at temperature T (K)."""
        props = {}
        T_clipped = np.clip(T, self.T_min, self.T_max)
        if T != T_clipped and config.LOG_LEVEL >= 1:
             # Use .item() if T is a 0-d array
             T_item = T.item() if isinstance(T, np.ndarray) else T
             T_clipped_item = T_clipped.item() if isinstance(T_clipped, np.ndarray) else T_clipped
             # print(f"Warning: Liq T {T_item:.1f}K outside range [{self.T_min:.1f}-{self.T_max:.1f}K]. Clipped to {T_clipped_item:.1f}K.")
             pass # Reduce verbosity

        for prop_name, interpolator in self.interpolators.items():
            props[prop_name] = float(interpolator(T_clipped))
        return props

    def get_prop(self, prop_name, T):
        """Returns a specific liquid property at temperature T (K)."""
        T_clipped = np.clip(T, self.T_min, self.T_max)
        if T != T_clipped and config.LOG_LEVEL >= 1:
             T_item = T.item() if isinstance(T, np.ndarray) else T
             T_clipped_item = T_clipped.item() if isinstance(T_clipped, np.ndarray) else T_clipped
             # print(f"Warning: Liq T {T_item:.1f}K outside range [{self.T_min:.1f}-{self.T_max:.1f}K]. Clipped to {T_clipped_item:.1f}K.")
             pass # Reduce verbosity

        if prop_name in self.interpolators:
            return float(self.interpolators[prop_name](T_clipped))
        else:
            valid_props = list(self.interpolators.keys())
            raise ValueError(f"Property '{prop_name}' not found in liquid data. Available: {valid_props}")


class GasProperties:
    """
    Wrapper around Cantera Gas object. Implements Redlich-Kwong EOS for density.
    Other properties are currently calculated using Cantera's ideal gas models.
    """
    def __init__(self, mech_file=config.MECH_FILE, use_rk=config.USE_RK_EOS):
        try:
            # Suppress Cantera warnings during init if possible (or handle them)
            with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", category=UserWarning, module="cantera")
                 self.gas = ct.Solution(mech_file)
        except Exception as e:
             raise RuntimeError(f"Failed to initialize Cantera Solution from {mech_file}: {e}")

        self.nsp = self.gas.n_species
        self.use_rk = use_rk
        try:
            self.fuel_idx = self.gas.species_index(config.FUEL_SPECIES_NAME)
        except ValueError:
             raise ValueError(f"Fuel species '{config.FUEL_SPECIES_NAME}' not found in mechanism '{mech_file}'")
        try:
            self.o2_idx = self.gas.species_index('O2')
            self.n2_idx = self.gas.species_index('N2')
        except ValueError as e:
             print(f"Warning: O2 or N2 not found in mechanism. Needed for some defaults/checks. Error: {e}")
             self.o2_idx = -1
             self.n2_idx = -1

        self.species_names = self.gas.species_names
        self.molecular_weights = self.gas.molecular_weights # kg/kmol

        if self.use_rk:
            print("Redlich-Kwong EOS enabled for density calculation.")
            self._init_rk_parameters()
        else:
             print("Using Cantera's Ideal Gas EOS for density calculation.")
             self.Tc = None; self.Pc = None; self.omega = None

    def _init_rk_parameters(self):
        """Load critical properties from config."""
        self.Tc = np.zeros(self.nsp)
        self.Pc = np.zeros(self.nsp)
        self.omega = np.zeros(self.nsp)
        loaded_count = 0
        print("Loading RK parameters from config...")
        for i, name in enumerate(self.species_names):
            if name in config.RK_PARAMS:
                params = config.RK_PARAMS[name]
                self.Tc[i] = params['Tc']
                self.Pc[i] = params['Pc']
                self.omega[i] = params['omega']
                loaded_count += 1
            else:
                # Set defaults (implies ideal gas for this species in RK calcs)
                self.Tc[i] = 0.0
                self.Pc[i] = 1.0e5 # Assign a dummy pressure > 0
                self.omega[i] = 0.0

        print(f"  Loaded RK parameters for {loaded_count}/{self.nsp} species from config.")
        if self.Tc[self.fuel_idx] <= 0.0:
             print(f"WARNING: RK parameters for Fuel '{config.FUEL_SPECIES_NAME}' missing in config! RK calcs inaccurate.")
        if self.o2_idx >=0 and self.Tc[self.o2_idx] <= 0.0: print("Warning: RK parameters for O2 missing.")
        if self.n2_idx >=0 and self.Tc[self.n2_idx] <= 0.0: print("Warning: RK parameters for N2 missing.")


    def set_state(self, T, P, Y):
        """Safely set the thermodynamic state of the Cantera object."""
        try:
            # Ensure T is within a reasonable range for Cantera
            T_safe = np.clip(T, 200.0, 5000.0) # Adjust range as needed
            if abs(T - T_safe) > 1e-2 and config.LOG_LEVEL >= 1:
                 print(f"Warning: Clipping temperature {T:.1f} K to {T_safe:.1f} K for Cantera.")

            Y_clean = np.maximum(Y, 0)
            sum_Y = np.sum(Y_clean)
            # Use a slightly looser tolerance for sum check? 1e-6 might be better.
            if abs(sum_Y - 1.0) > 1e-6:
                if sum_Y > 1e-6:
                    Y_clean /= sum_Y
                else: # All fractions are zero
                    if config.LOG_LEVEL >= 1: print(f"Warning: Mass fractions sum to {sum_Y:.1e} at T={T:.1f}, P={P:.1e}. Resetting to N2.")
                    Y_clean = np.zeros_like(Y)
                    if self.n2_idx >= 0: Y_clean[self.n2_idx] = 1.0
                    else: Y_clean[0] = 1.0 # Fallback

            self.gas.TPY = T_safe, P, Y_clean
            # Store last valid state
            self.last_T = T_safe
            self.last_P = P
            self.last_Y = Y_clean.copy()
            return True
        except (ct.CanteraError, ValueError) as e:
            if config.LOG_LEVEL >= 0: # Make this warning less frequent?
                print(f"Warning: Cantera Error setting state T={T:.1f}K, P={P:.1e}Pa: {e}")
            return False

    def _rk_eos_residual(self, Z, A_mix, B_mix):
        """Residual form of the RK cubic equation for the solver."""
        return Z**3 - Z**2 + (A_mix - B_mix - B_mix**2) * Z - A_mix * B_mix

    def _solve_rk_eos(self, T, P, X):
        """Solves the Redlich-Kwong EOS for compressibility factor Z using Newton's method."""
        R_univ = config.R_UNIVERSAL # J/(mol K)

        # 1. Calculate pure component parameters a_i, b_i, alpha_i
        # Avoid division by zero or sqrt of zero/negative for species without RK params
        valid_rk_mask = (self.Tc > 0) & (self.Pc > 0)
        Tci = np.where(valid_rk_mask, self.Tc, 1.0) # Use dummy 1.0 K
        Pci = np.where(valid_rk_mask, self.Pc, 1.0e5) # Use dummy 1 Pa
        omegai = np.where(valid_rk_mask, self.omega, 0.0)
        Tr_i = T / Tci # Reduced temperature

        # RK parameter 'a' depends on T^-0.5
        a_i = np.where(valid_rk_mask, 0.42748 * (R_univ**2 * Tci**2.5) / Pci, 0.0)
        b_i = np.where(valid_rk_mask, 0.08664 * (R_univ * Tci) / Pci, 0.0)
        # alpha_i = 1.0 / np.sqrt(Tr_i) # Original RK alpha
        # Soave modification (SRK) - Let's use the standard RK T^-0.5 dependence for a_i
        # a_i_alpha_i = a_i / np.sqrt(Tr_i) ? No, definition of a includes T^2.5
        # Let's redefine a and alpha for clarity based on common formulations:
        # a_i_const = 0.42748 * (R_univ**2 * Tci**2) / Pci
        # alpha_T_i = 1.0 / np.sqrt(Tr_i) # Temperature dependent part for RK
        # a_i_eff = a_i_const * alpha_T_i

        # 2. Apply mixing rules (Van der Waals one-fluid)
        # Use X (mole fractions)
        a_mix = 0.0
        b_mix = 0.0
        for i in range(self.nsp):
             b_mix += X[i] * b_i[i]
             for j in range(self.nsp):
                 # Calculate effective a_i including temperature dependence
                 a_i_eff = a_i[i] / np.sqrt(T) if Tci[i]>0 else 0.0
                 a_j_eff = a_i[j] / np.sqrt(T) if Tci[j]>0 else 0.0
                 # Check if sqrt becomes complex? Should be ok if T>0
                 a_mix += X[i] * X[j] * np.sqrt(a_i_eff * a_j_eff) # Geometric mean for cross term

        # 3. Dimensionless parameters for the cubic equation
        # Ensure T is not zero
        T_safe = max(T, 1e-2)
        A_mix = a_mix * P / (R_univ**2 * T_safe**2) # Note: a_mix already includes T^-0.5 * T^-0.5
        B_mix = b_mix * P / (R_univ * T_safe)

        # 4. Solve the cubic equation for Z: Z^3 - Z^2 + (A_mix - B_mix - B_mix^2)Z - A_mix*B_mix = 0
        # Use an iterative solver like Newton's method for robustness
        # Initial guess: Ideal gas Z=1
        try:
            # Provide residual function to Newton solver
            Z_solution = newton(self._rk_eos_residual, x0=1.0, args=(A_mix, B_mix), tol=config.RK_SOLVER_TOL, maxiter=config.MAX_ITER_RK)
            # Check if the solution is physical (Z > B_mix)
            if Z_solution > B_mix:
                 Z = Z_solution
            else: # If not physical, try to find liquid root (smallest positive root > B?) - unlikely needed for gas
                 # Alternative: Use np.roots and select physical root
                 coeffs = [1.0, -1.0, A_mix - B_mix - B_mix**2, -A_mix * B_mix]
                 roots = np.roots(coeffs)
                 real_roots = roots[np.isreal(roots)].real
                 physical_roots = real_roots[real_roots > B_mix]
                 if len(physical_roots) > 0:
                     Z = np.max(physical_roots) # Vapor root
                 else:
                     Z = 1.0 # Fallback if no physical root found
                     if config.LOG_LEVEL >= 1: print(f"Warning: No physical RK root found (Z>{B_mix:.2e}) at T={T:.1f}, P={P:.1e}. Using Z=1.")

        except (RuntimeError, OverflowError): # Solver failed to converge or overflow
            Z = 1.0 # Fallback to ideal gas
            if config.LOG_LEVEL >= 1: print(f"Warning: RK EOS solver failed at T={T:.1f}, P={P:.1e}. Using Z=1.")

        # Final safety check
        if np.isnan(Z) or Z <= 0:
             Z = 1.0

        return Z

    def get_density(self, T, P, Y):
        """Calculate density using Ideal Gas or Redlich-Kwong."""
        if not self.set_state(T, P, Y):
            # Use last valid state if current fails
            if hasattr(self, 'last_T') and self.set_state(self.last_T, self.last_P, self.last_Y):
                 if config.LOG_LEVEL >=1: print(f"Info: Using last valid state for density T={self.last_T:.1f} P={self.last_P:.1e}")
            else:
                 print("Error: Cannot set state in get_density, no fallback available.")
                 return np.nan

        if self.use_rk and np.any(self.Tc > 0): # Check if ANY critical params are loaded
            try:
                X = self.gas.X # Mole fractions
                # Check for negative mole fractions which can cause sqrt issues
                if np.any(X < -1e-9):
                     if config.LOG_LEVEL >= 1: print(f"Warning: Negative mole fractions detected in get_density: {X[X<-1e-9]}. Clipping.")
                     X = np.maximum(X, 0.0)
                     X /= np.sum(X) # Renormalize

                Z = self._solve_rk_eos(self.gas.T, self.gas.P, X) # Use T, P from gas object
                mean_mw_kmol = self.gas.mean_molecular_weight # kg/kmol
                R_kmol = config.R_UNIVERSAL * 1000.0 # J/(kmol*K)
                density_rk = self.gas.P * mean_mw_kmol / (Z * R_kmol * self.gas.T)
                if np.isnan(density_rk) or density_rk <= 0:
                    if config.LOG_LEVEL >= 1: print(f"Warning: RK density calculation resulted in {density_rk}. Using ideal gas.")
                    return self.gas.density # Fallback to ideal gas
                return density_rk
            except Exception as e:
                print(f"Error during RK density calculation at T={self.gas.T:.1f}, P={self.gas.P:.1e}: {e}. Falling back to ideal gas.")
                return self.gas.density
        else:
            return self.gas.density

    # Other properties use Cantera's ideal gas values
    def get_cp_mass(self, T, P, Y): # Renamed for clarity
        """Calculate mass specific heat cp."""
        if not self.set_state(T, P, Y): return np.nan
        return self.gas.cp_mass # J/kg/K

    def get_enthalpy_mass(self, T, P, Y):
         """Calculate mass specific enthalpy."""
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.enthalpy_mass # J/kg

    def get_partial_enthalpies_mass(self, T, P, Y):
         """Calculate mass specific partial enthalpies."""
         if not self.set_state(T, P, Y): return np.full(self.nsp, np.nan)
         Hi_molar = self.gas.partial_molar_enthalpies # J/kmol
         Mi_kg_kmol = self.molecular_weights # kg/kmol
         # Avoid division by zero if MW is zero (should not happen)
         Mi_safe = np.maximum(Mi_kg_kmol, 1e-6)
         return Hi_molar / Mi_safe # J/kg

    def get_thermal_conductivity(self, T, P, Y):
         """Get thermal conductivity."""
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.thermal_conductivity # W/m/K

    def get_viscosity(self, T, P, Y):
         """Get viscosity."""
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.viscosity # Pa*s

    def get_diffusion_coeffs(self, T, P, Y, option=config.DIFFUSION_OPTION):
         """
         Get diffusion coefficients based on the selected option.
         Returns an array Dk for each species [m^2/s].
         """
         if not self.set_state(T, P, Y): return np.full(self.nsp, np.nan)

         if option == 'constant':
             return np.full(self.nsp, config.DIFFUSION_CONSTANT_VALUE)

         rho = self.get_density(T, P, Y) # Use consistent density
         cp_mass = self.gas.cp_mass
         lambda_val = self.gas.thermal_conductivity

         if np.isnan(rho) or np.isnan(cp_mass) or np.isnan(lambda_val):
             print(f"Warning: NaN properties encountered in get_diffusion_coeffs (T={T:.1f}).")
             return np.full(self.nsp, np.nan)

         if option == 'Le=1':
             if rho > 1e-6 and cp_mass > 1e-6:
                 D_eff = lambda_val / (rho * cp_mass)
             else:
                 D_eff = 1e-12 # Avoid division by zero, use very small value
             return np.full(self.nsp, max(D_eff, 1e-12)) # Ensure non-negative
         elif option == 'mixture_averaged':
             try:
                 # This gives mole-fraction based mixture-averaged diffusion coefficients
                 Dk_mix_mole = self.gas.mix_diff_coeffs # m^2/s
                 # Need conversion to mass-averaged or effective D for Fick's law in mass form?
                 # For dYk/dt = Div(rho*Dk*grad(Yk)), Dk here is the effective mass diffusivity.
                 # Often Dk_eff ~ Dk_mix_mole is used as an approximation, or relate via Le=1.
                 # Let's use Dk_mix_mole directly as an approximation for Dk_eff
                 # print("Using Cantera's mix_diff_coeffs directly.")
                 return np.maximum(Dk_mix_mole, 1e-12) # Ensure non-negative
             except Exception as e:
                 print(f"Warning: Failed to get mixture_averaged diff coeffs: {e}. Falling back to Le=1.")
                 if rho > 1e-6 and cp_mass > 1e-6: D_eff = lambda_val / (rho * cp_mass)
                 else: D_eff = 1e-12
                 return np.full(self.nsp, max(D_eff, 1e-12))
         else:
             raise ValueError(f"Unknown diffusion option: {option}")