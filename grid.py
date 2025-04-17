# =========================================
#            grid.py (修正後)
# =========================================
# grid.py
"""Functions for generating liquid and gas phase grids for FVM."""

import numpy as np
import config

def liquid_grid_fvm(R, Nl):
    """
    Generates the liquid phase grid for Finite Volume Method (FVM).
    Returns cell center locations, face locations, and cell volumes.
    Cell 0 is at the center, Cell Nl-1 is at the surface.
    Faces: Nl+1 locations (node 0 at r=0, node Nl at r=R)
    Centers: Nl locations
    Volumes: Nl volumes
    """
    if Nl <= 0:
        return np.array([]), np.array([]), np.array([])
    if Nl == 1: # Single cell covering the whole sphere
        r_nodes = np.array([0.0, R]) # Face locations
        r_centers = np.array([0.5 * R]) # Cell center (approx)
        volumes = np.array([(4.0/3.0) * np.pi * R**3])
        return r_centers, r_nodes, volumes

    # Generate Nl+1 face locations uniformly from 0 to R
    r_nodes = np.linspace(0, R, Nl + 1)

    # Calculate cell center locations (midpoints between faces)
    r_centers = 0.5 * (r_nodes[:-1] + r_nodes[1:])

    # Calculate cell volumes (volume of spherical shell)
    volumes = (4.0 / 3.0) * np.pi * (r_nodes[1:]**3 - r_nodes[:-1]**3)

    # Define center cell node at r=0 (for convenience in indexing/plotting)
    # Note: The first volume calculation is correct for the sphere from 0 to r_nodes[1]
    if Nl > 0:
        r_centers[0] = 0.0

    return r_centers, r_nodes, volumes

def gas_grid_fvm(R, rmax, Ng):
    """
    Generates the gas phase grid for FVM using a geometric progression.
    Cell 0 is adjacent to the droplet surface, Cell Ng-1 is at rmax.
    Returns cell center locations, face locations, and cell volumes.
    Faces: Ng+1 locations (node 0 at r=R, node Ng at r=rmax)
    Centers: Ng locations
    Volumes: Ng volumes
    """
    if Ng <= 0:
        return np.array([]), np.array([]), np.array([])
    if Ng == 1: # Single cell covering the whole gas domain
        r_nodes = np.array([R, rmax]) # Face locations
        r_centers = np.array([0.5 * (R + rmax)]) # Cell center (approx)
        volumes = np.array([(4.0/3.0) * np.pi * (rmax**3 - R**3)])
        return r_centers, r_nodes, volumes

    # Generate Ng+1 face locations using geometric progression
    g_nodes = np.arange(Ng + 1)
    exponent_nodes = g_nodes / Ng # Exponent from 0 to 1

    # Prevent numerical issues if R is extremely small or rmax=R
    if R < 1e-15: R = 1e-15
    ratio = rmax / R
    if ratio <= 1.0: ratio = 1.0 + 1e-6 # Ensure ratio > 1
    if config.GRID_TYPE == 'geometric_series':
        r_nodes = R * ratio**exponent_nodes
    elif config.GRID_TYPE == 'power_law':
        xi = config.XI_GAS_GRID
        r_nodes = R + (rmax-R)*exponent_nodes**xi
    # Ensure boundaries are exact
    r_nodes[0] = R
    r_nodes[-1] = rmax

    # Calculate cell center locations (Geometric mean might be better for log spacing)
    # r_centers = np.sqrt(r_nodes[:-1] * r_nodes[1:]) # Geometric mean
    r_centers = 0.5 * (r_nodes[:-1] + r_nodes[1:]) # Arithmetic mean (simpler)

    # Calculate cell volumes (volume of spherical shell)
    volumes = (4.0 / 3.0) * np.pi * (r_nodes[1:]**3 - r_nodes[:-1]**3)
    # Ensure volumes are non-negative
    volumes = np.maximum(volumes, 1e-30)

    return r_centers, r_nodes, volumes

# Helper function to get face areas (needed for flux calculations)
def face_areas(r_nodes):
    """Calculates the area of the faces defined by r_nodes."""
    if len(r_nodes) < 1:
        return np.array([])
    return 4.0 * np.pi * r_nodes**2