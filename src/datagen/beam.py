"""Beam parameter sampling and particle generation via distgen.

Samples Twiss parameters, emittances, energy spread, bunch length,
and centroid offsets, then generates particles using distgen.
"""

import numpy as np
from pathlib import Path
from distgen import Generator

# --- Beam parameter ranges (from DATA_GENERATION.md) ---

BETA_RANGE = (0.5, 100.0)          # m, log-sampled
ALPHA_RANGE = (-3.0, 3.0)          # dimensionless
EMIT_RANGE = (1e-10, 1e-5)         # m*rad (0.1 nm to 10 um), log-sampled
SIGMA_DELTA_RANGE = (1e-4, 1e-2)   # relative, log-sampled
SIGMA_Z_RANGE = (1e-4, 0.1)        # m (0.1 mm to 10 cm), log-sampled
ENERGY_RANGE = (0.5, 10.0)         # GeV, log-sampled
CENTROID_CLIP = 3.0                 # sigma units


def sample_beam_params(rng: np.random.Generator) -> dict:
    """Sample random beam parameters.

    Returns:
        Dictionary with keys: beta_x, beta_y, alpha_x, alpha_y,
        emit_x, emit_y, sigma_delta, sigma_z, energy_GeV,
        offset_x, offset_px, offset_y, offset_py, offset_z, offset_delta.
    """
    params = {}

    # Twiss
    params['beta_x'] = np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1])))
    params['beta_y'] = np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1])))
    params['alpha_x'] = rng.uniform(*ALPHA_RANGE)
    params['alpha_y'] = rng.uniform(*ALPHA_RANGE)

    # Emittance (geometric)
    params['emit_x'] = np.exp(rng.uniform(np.log(EMIT_RANGE[0]), np.log(EMIT_RANGE[1])))
    params['emit_y'] = np.exp(rng.uniform(np.log(EMIT_RANGE[0]), np.log(EMIT_RANGE[1])))

    # Longitudinal
    params['sigma_delta'] = np.exp(rng.uniform(
        np.log(SIGMA_DELTA_RANGE[0]), np.log(SIGMA_DELTA_RANGE[1])
    ))
    params['sigma_z'] = np.exp(rng.uniform(
        np.log(SIGMA_Z_RANGE[0]), np.log(SIGMA_Z_RANGE[1])
    ))

    # Reference energy
    params['energy_GeV'] = np.exp(rng.uniform(
        np.log(ENERGY_RANGE[0]), np.log(ENERGY_RANGE[1])
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
        'total_charge': {'value': 0, 'units': 'C'},
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

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        P.write(str(output_path))

    return P
