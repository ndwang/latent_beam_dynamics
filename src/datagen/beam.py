"""Beam parameter sampling and particle generation via distgen.

Samples Twiss parameters, emittances, energy spread, bunch length,
and centroid offsets, then generates particles using distgen.
"""

import numpy as np
from pathlib import Path
from distgen import Generator

# --- Beam parameter ranges ---

BETA_RANGE = (1.0, 30.0)           # m, log-sampled
ALPHA_RANGE = (-2.0, 2.0)          # dimensionless
EMIT_N_RANGE = (1e-7, 1e-4)        # m*rad normalized (0.1 um to 100 um), log-sampled
SIGMA_DELTA_RANGE = (1e-4, 1e-2)   # relative, log-sampled
SIGMA_Z_RANGE = (1e-4, 0.1)        # m (0.1 mm to 10 cm), log-sampled
ENERGY_RANGE = (0.1, 20.0)         # GeV, log-sampled
CENTROID_CLIP = 3.0                 # sigma units
ELECTRON_MASS_EV = 0.510998950e6   # electron rest mass in eV

# Mismatch parameter range for matched beam sampling
BMAG_RANGE = (1.0, 10.0)           # log-sampled


def sample_beam_params(rng: np.random.Generator) -> dict:
    """Sample random beam parameters (unmatched).

    Returns:
        Dictionary with keys: beta_x, beta_y, alpha_x, alpha_y,
        emit_x, emit_y, sigma_delta, sigma_z, energy_GeV,
        offset_x, offset_px, offset_y, offset_py, offset_z, offset_delta.
    """
    params = {}

    # Reference energy (sampled first — needed for emittance conversion)
    params['energy_GeV'] = np.exp(rng.uniform(
        np.log(ENERGY_RANGE[0]), np.log(ENERGY_RANGE[1])
    ))

    # Twiss
    params['beta_x'] = np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1])))
    params['beta_y'] = np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1])))
    params['alpha_x'] = rng.uniform(*ALPHA_RANGE)
    params['alpha_y'] = rng.uniform(*ALPHA_RANGE)

    # Normalized emittance -> geometric
    beta_gamma = params['energy_GeV'] * 1e9 / ELECTRON_MASS_EV
    emit_n_x = np.exp(rng.uniform(np.log(EMIT_N_RANGE[0]), np.log(EMIT_N_RANGE[1])))
    emit_n_y = np.exp(rng.uniform(np.log(EMIT_N_RANGE[0]), np.log(EMIT_N_RANGE[1])))
    params['emit_n_x'] = emit_n_x
    params['emit_n_y'] = emit_n_y
    params['emit_x'] = emit_n_x / beta_gamma
    params['emit_y'] = emit_n_y / beta_gamma

    # Longitudinal
    params['sigma_delta'] = np.exp(rng.uniform(
        np.log(SIGMA_DELTA_RANGE[0]), np.log(SIGMA_DELTA_RANGE[1])
    ))
    params['sigma_z'] = np.exp(rng.uniform(
        np.log(SIGMA_Z_RANGE[0]), np.log(SIGMA_Z_RANGE[1])
    ))

    # Centroid offsets in units of sigma, clipped to [-3, 3]
    for coord in ['x', 'px', 'y', 'py', 'z', 'delta']:
        params[f'offset_{coord}'] = np.clip(rng.normal(), -CENTROID_CLIP, CENTROID_CLIP)

    return params


def sample_matched_beam_params(
    rng: np.random.Generator,
    lattice_info: dict,
) -> dict:
    """Sample beam parameters approximately matched to a sectioned lattice.

    Uses the periodic Twiss of the first section as a reference and applies
    a controlled mismatch via the B_mag parameter. At B_mag=1 the beam is
    perfectly matched; at B_mag=5 the envelope oscillates with ~5x amplitude.

    The initial alpha is set to 0 (symmetric about quad center) with a small
    random perturbation.

    Args:
        rng: NumPy random generator.
        lattice_info: Dictionary from sample_sectioned_lattice() containing
            beta_x_periodic, beta_y_periodic, and section metadata.

    Returns:
        Beam parameter dictionary (same format as sample_beam_params).
    """
    params = {}

    # Reference energy
    params['energy_GeV'] = np.exp(rng.uniform(
        np.log(ENERGY_RANGE[0]), np.log(ENERGY_RANGE[1])
    ))

    # Mismatch parameter (log-uniform for even coverage of 1x to 5x)
    bmag = np.exp(rng.uniform(np.log(BMAG_RANGE[0]), np.log(BMAG_RANGE[1])))
    params['bmag'] = float(bmag)

    # Twiss matched to lattice with controlled mismatch
    # At a QF center: beta_x = beta_max, beta_y = beta_min, alpha = 0
    # Mismatch: scale beta by bmag in one plane and 1/bmag in the other
    # to get diverse but bounded dynamics
    beta_x_ref = lattice_info["beta_x_periodic"]
    beta_y_ref = lattice_info["beta_y_periodic"]

    # Randomly assign which plane gets the larger beta
    if rng.random() < 0.5:
        params['beta_x'] = beta_x_ref * bmag
        params['beta_y'] = beta_y_ref / bmag
    else:
        params['beta_x'] = beta_x_ref / bmag
        params['beta_y'] = beta_y_ref * bmag

    # Small alpha perturbation (matched alpha = 0 at quad center)
    params['alpha_x'] = rng.normal(0, 0.3)
    params['alpha_y'] = rng.normal(0, 0.3)

    # Normalized emittance -> geometric (same ranges as unmatched)
    beta_gamma = params['energy_GeV'] * 1e9 / ELECTRON_MASS_EV
    emit_n_x = np.exp(rng.uniform(np.log(EMIT_N_RANGE[0]), np.log(EMIT_N_RANGE[1])))
    emit_n_y = np.exp(rng.uniform(np.log(EMIT_N_RANGE[0]), np.log(EMIT_N_RANGE[1])))
    params['emit_n_x'] = emit_n_x
    params['emit_n_y'] = emit_n_y
    params['emit_x'] = emit_n_x / beta_gamma
    params['emit_y'] = emit_n_y / beta_gamma

    # Longitudinal
    params['sigma_delta'] = np.exp(rng.uniform(
        np.log(SIGMA_DELTA_RANGE[0]), np.log(SIGMA_DELTA_RANGE[1])
    ))
    params['sigma_z'] = np.exp(rng.uniform(
        np.log(SIGMA_Z_RANGE[0]), np.log(SIGMA_Z_RANGE[1])
    ))

    # Centroid offsets in units of sigma, clipped to [-3, 3]
    for coord in ['x', 'px', 'y', 'py', 'z', 'delta']:
        params[f'offset_{coord}'] = np.clip(rng.normal(), -CENTROID_CLIP, CENTROID_CLIP)

    return params


def _beam_sizes(params: dict) -> dict:
    """Compute RMS beam sizes from Twiss and emittance.

    Returns:
        Dictionary with sigma_x, sigma_px, sigma_y, sigma_py, sigma_z,
        sigma_pz (all in SI: meters and eV/c).
    """
    emit_x = params['emit_x']
    emit_y = params['emit_y']
    beta_x = params['beta_x']
    beta_y = params['beta_y']
    alpha_x = params['alpha_x']
    alpha_y = params['alpha_y']

    gamma_x = (1 + alpha_x**2) / beta_x
    gamma_y = (1 + alpha_y**2) / beta_y

    sigma_x = np.sqrt(emit_x * beta_x)
    sigma_y = np.sqrt(emit_y * beta_y)
    sigma_px_norm = np.sqrt(emit_x * gamma_x)  # in rad (x' = px/p)
    sigma_py_norm = np.sqrt(emit_y * gamma_y)

    # Convert angular divergence to momentum: px [eV/c] = x' * p [eV/c]
    p_eV = params['energy_GeV'] * 1e9  # eV/c (ultrarelativistic)
    sigma_px = sigma_px_norm * p_eV
    sigma_py = sigma_py_norm * p_eV

    sigma_z = params['sigma_z']
    sigma_pz = params['sigma_delta'] * p_eV

    return {
        'sigma_x': sigma_x,
        'sigma_px': sigma_px,
        'sigma_y': sigma_y,
        'sigma_py': sigma_py,
        'sigma_z': sigma_z,
        'sigma_pz': sigma_pz,
        'p_eV': p_eV,
    }


def generate_particles(
    params: dict,
    n_particles: int = 100_000,
    output_path: str | Path | None = None,
) -> 'ParticleGroup':
    """Generate particles using distgen and apply Twiss transform.

    1. Generate a round Gaussian beam with unit Twiss (beta=1, alpha=0).
    2. Apply set_twiss transform to reach desired Twiss/emittance.
    3. Apply centroid offsets via translate transform.

    Args:
        params: Beam parameter dictionary from sample_beam_params().
        n_particles: Number of macro-particles.
        output_path: If provided, write particles to HDF5.

    Returns:
        ParticleGroup with generated particles.
    """
    sizes = _beam_sizes(params)

    distgen_input = {
        'n_particle': n_particles,
        'species': 'electron',
        'total_charge': {'value': 1e-9, 'units': 'C'},
        'random': {'type': 'hammersley'},
        'start': {
            'type': 'time',
            'tstart': {'value': 0, 'units': 's'},
        },
        'x_dist': {
            'type': 'gaussian',
            'sigma_x': {'value': float(sizes['sigma_x']), 'units': 'm'},
            'avg_x': {'value': 0, 'units': 'm'},
        },
        'px_dist': {
            'type': 'gaussian',
            'sigma_px': {'value': float(sizes['sigma_px']), 'units': 'eV/c'},
            'avg_px': {'value': 0, 'units': 'eV/c'},
        },
        'y_dist': {
            'type': 'gaussian',
            'sigma_y': {'value': float(sizes['sigma_y']), 'units': 'm'},
            'avg_y': {'value': 0, 'units': 'm'},
        },
        'py_dist': {
            'type': 'gaussian',
            'sigma_py': {'value': float(sizes['sigma_py']), 'units': 'eV/c'},
            'avg_py': {'value': 0, 'units': 'eV/c'},
        },
        'z_dist': {
            'type': 'gaussian',
            'sigma_z': {'value': float(sizes['sigma_z']), 'units': 'm'},
            'avg_z': {'value': 0, 'units': 'm'},
        },
        'pz_dist': {
            'type': 'gaussian',
            'sigma_pz': {'value': float(sizes['sigma_pz']), 'units': 'eV/c'},
            'avg_pz': {'value': float(sizes['p_eV']), 'units': 'eV/c'},
        },
        'transforms': {
            'set_twiss_x': {
                'type': 'set_twiss x',
                'beta': {'value': float(params['beta_x']), 'units': 'm'},
                'alpha': {'value': float(params['alpha_x']), 'units': ''},
                'emittance': {'value': float(params['emit_x']), 'units': 'm'},
            },
            'set_twiss_y': {
                'type': 'set_twiss y',
                'beta': {'value': float(params['beta_y']), 'units': 'm'},
                'alpha': {'value': float(params['alpha_y']), 'units': ''},
                'emittance': {'value': float(params['emit_y']), 'units': 'm'},
            },
        },
    }

    # Add centroid offset transforms (one per coordinate)
    offset_map = {
        'offset_x':     ('x',  sizes['sigma_x'],  'm'),
        'offset_px':    ('px', sizes['sigma_px'],  'eV/c'),
        'offset_y':     ('y',  sizes['sigma_y'],  'm'),
        'offset_py':    ('py', sizes['sigma_py'],  'eV/c'),
        'offset_z':     ('z',  sizes['sigma_z'],  'm'),
        'offset_delta': ('pz', sizes['sigma_pz'],  'eV/c'),
    }
    for key, (var, sigma, units) in offset_map.items():
        if params[key] != 0:
            distgen_input['transforms'][f'translate_{var}'] = {
                'type': f'translate {var}',
                'delta': {'value': float(params[key] * sigma), 'units': units},
            }

    G = Generator(distgen_input)
    G.run()
    P = G.particles

    # Convert z -> t so Bmad sees the longitudinal spread.
    # Bmad uses t (arrival time) not z at a fixed s-position.
    # t = -z / (beta * c), then zero out z.
    c = 299792458.0
    P.t = -P.z / (P.beta * c)
    P.z = np.zeros_like(P.z)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        P.write(str(output_path))

    return P
