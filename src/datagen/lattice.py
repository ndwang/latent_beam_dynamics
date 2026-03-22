"""Lattice sampling and Bmad file generation.

Element parameter vector: [L, K1, K2, Angle, V_rf, f_rf, phi_rf]
Indices:                    0   1   2    3      4     5     6
"""

import numpy as np
from pathlib import Path

# Element parameter indices
I_L = 0
I_K1 = 1
I_K2 = 2
I_ANGLE = 3
I_VRF = 4
I_FRF = 5
I_PHIRF = 6

ELEMENT_DIM = 7

# --- Parameter ranges (from DATA_GENERATION.md) ---

L_DRIFT = (0.2, 5.0)       # m
L_QUAD = (0.1, 0.5)        # m
L_SEXT = (0.1, 0.3)        # m
L_DIPOLE = (0.5, 3.0)      # m
L_RF = (0.1, 2.0)          # m

K1_RANGE = (0.5, 5.0)      # m^-2 (absolute value)
K2_RANGE = (1.0, 10.0)     # m^-3 (absolute value)
ANGLE_RANGE = (0.01, 0.5)  # rad
VRF_RANGE = (0.1, 10.0)    # MV
FRF_RANGE = (0.1, 3.0)     # GHz
PHIRF_RANGE = (-np.pi / 6, np.pi / 6)  # rad


def _sample_drift(rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = rng.uniform(*L_DRIFT)
    return e


def _sample_quad(rng: np.random.Generator, sign: int = 0) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = rng.uniform(*L_QUAD)
    k1_mag = rng.uniform(*K1_RANGE)
    if sign == 0:
        sign = rng.choice([-1, 1])
    e[I_K1] = sign * k1_mag
    return e


def _sample_sextupole(rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = rng.uniform(*L_SEXT)
    e[I_K2] = rng.uniform(*K2_RANGE) * rng.choice([-1, 1])
    return e


def _sample_dipole(rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = rng.uniform(*L_DIPOLE)
    e[I_ANGLE] = rng.uniform(*ANGLE_RANGE)
    return e


def _sample_rf(rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = rng.uniform(*L_RF)
    e[I_VRF] = np.exp(rng.uniform(np.log(VRF_RANGE[0]), np.log(VRF_RANGE[1])))
    e[I_FRF] = np.exp(rng.uniform(np.log(FRF_RANGE[0]), np.log(FRF_RANGE[1])))
    e[I_PHIRF] = rng.uniform(*PHIRF_RANGE)
    return e


# --- Structured lattice ---

def sample_structured_lattice(seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a structured lattice with alternating-quad backbone.

    Layout: elements are built as quad-drift pairs with optional
    dipoles, RF cavities, or sextupoles inserted between quads.
    All non-drift elements are separated by drifts.

    Args:
        seq_len: Number of elements in the lattice.
        rng: NumPy random generator.

    Returns:
        (seq_len, 7) array of element parameters.
    """
    elements = []
    quad_sign = rng.choice([-1, 1])  # starting sign

    while len(elements) < seq_len:
        # Quad
        elements.append(_sample_quad(rng, sign=quad_sign))
        quad_sign *= -1
        if len(elements) >= seq_len:
            break

        # Drift after quad
        elements.append(_sample_drift(rng))
        if len(elements) >= seq_len:
            break

        # Optionally insert extra elements before next quad
        r = rng.random()
        if r < 0.2:
            # Insert dipole + drift
            elements.append(_sample_dipole(rng))
            if len(elements) >= seq_len:
                break
            elements.append(_sample_drift(rng))
        elif r < 0.35:
            # Insert RF + drift
            elements.append(_sample_rf(rng))
            if len(elements) >= seq_len:
                break
            elements.append(_sample_drift(rng))
        elif r < 0.45:
            # Insert sextupole + drift
            elements.append(_sample_sextupole(rng))
            if len(elements) >= seq_len:
                break
            elements.append(_sample_drift(rng))
        # else: nothing extra, next iteration adds quad

    elements = elements[:seq_len]
    return np.array(elements)


# --- Random lattice ---

# Type probabilities: drift 40%, quad 30%, dipole 15%, sextupole 10%, RF 5%
_RANDOM_TYPES = ['drift', 'quad', 'dipole', 'sextupole', 'rf']
_RANDOM_PROBS = [0.40, 0.30, 0.15, 0.10, 0.05]

_SAMPLERS = {
    'drift': _sample_drift,
    'quad': _sample_quad,
    'dipole': _sample_dipole,
    'sextupole': _sample_sextupole,
    'rf': _sample_rf,
}


def sample_random_lattice(seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a fully random lattice.

    Element types are sampled independently per slot with fixed
    probabilities. No structural constraints.

    Args:
        seq_len: Number of elements in the lattice.
        rng: NumPy random generator.

    Returns:
        (seq_len, 7) array of element parameters.
    """
    types = rng.choice(_RANDOM_TYPES, size=seq_len, p=_RANDOM_PROBS)
    elements = np.zeros((seq_len, ELEMENT_DIM))
    for i, t in enumerate(types):
        elements[i] = _SAMPLERS[t](rng)
    return elements


# --- Bmad file writer ---

def write_bmad_lattice(
    element_params: np.ndarray,
    energy_GeV: float,
    output_path: str | Path,
) -> None:
    """Write a Bmad lattice file from element parameters.

    Args:
        element_params: (seq_len, 7) element parameter array.
        energy_GeV: Reference energy in GeV.
        output_path: Path to write the .bmad file.
    """
    output_path = Path(output_path)
    lines = []
    names = []

    for i, e in enumerate(element_params):
        L = e[I_L]
        K1 = e[I_K1]
        K2 = e[I_K2]
        angle = e[I_ANGLE]
        V_rf = e[I_VRF]
        f_rf = e[I_FRF]
        phi_rf = e[I_PHIRF]

        # Determine element type from nonzero parameters
        if V_rf != 0:
            name = f"rf_{i:04d}"
            # V_rf in MV -> V in volts, f_rf in GHz -> Hz
            lines.append(
                f"{name}: rfcavity, l = {L:.6f}"
                f", voltage = {V_rf * 1e6:.6f}"
                f", rf_frequency = {f_rf * 1e9:.6f}"
                f", phi0 = {phi_rf:.6f}"
            )
        elif angle != 0:
            name = f"b_{i:04d}"
            lines.append(
                f"{name}: sbend, l = {L:.6f}, angle = {angle:.6f}"
            )
        elif K2 != 0:
            name = f"s_{i:04d}"
            lines.append(
                f"{name}: sextupole, l = {L:.6f}, k2 = {K2:.6f}"
            )
        elif K1 != 0:
            name = f"q_{i:04d}"
            lines.append(
                f"{name}: quadrupole, l = {L:.6f}, k1 = {K1:.6f}"
            )
        else:
            name = f"d_{i:04d}"
            lines.append(f"{name}: drift, l = {L:.6f}")

        names.append(name)

    # Lattice line
    line_def = "lat: line = (" + ", ".join(names) + ")"
    lines.append(line_def)
    lines.append("use, lat")

    # Global parameters
    energy_eV = energy_GeV * 1e9
    lines.append(f"beginning[e_tot] = {energy_eV:.6f}")
    lines.append("beginning[beta_a] = 1")
    lines.append("beginning[beta_b] = 1")
    lines.append("parameter[particle] = electron")
    lines.append("parameter[geometry] = open")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
