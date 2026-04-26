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

# --- Parameter ranges ---

L_DRIFT = (0.2, 2.0)       # m
L_QUAD = (0.1, 0.5)        # m
L_SEXT = (0.1, 0.3)        # m
L_DIPOLE = (0.5, 3.0)      # m
L_RF = (0.1, 2.0)          # m

# Integrated strengths (controls total kick independent of length)
K1L_RANGE = (0.05, 1.0)    # m^-1 (focal length f = 1/K1L >= 1 m)
K2L_RANGE = (0.1, 1.5)     # m^-2
ANGLE_RANGE = (0.01, 0.15) # rad
VRF_RANGE = (0.01, 20.0)   # MV
FRF_RANGE = (0.1, 3.0)     # GHz
PHIRF_RANGE = (-np.pi / 6, np.pi / 6)  # rad (±30°, near stable fixed point)

# --- Sectioned lattice ranges ---

MU_RANGE = (np.radians(10), np.radians(85))  # phase advance per cell
L_HALF_CELL = (0.5, 5.0)    # half-cell drift-equivalent length (m)
SIGMA_PERT_RANGE = (0.0, 0.40)  # per-cell perturbation amplitude
P_RF_RANGE = (0.0, 0.3)     # probability of RF in straight half-cells
P_SEXT_RANGE = (0.0, 0.3)   # probability of sextupole in arc half-cells


# ==========================================================================
# Element constructors (fixed parameters)
# ==========================================================================

def _make_drift(L: float) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = L
    return e


def _make_quad(L: float, K1: float) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = L
    e[I_K1] = K1
    return e


def _make_bend(L: float, angle: float) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = L
    e[I_ANGLE] = angle
    return e


def _make_sextupole(L: float, K2: float) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = L
    e[I_K2] = K2
    return e


def _make_rf(L: float, rng: np.random.Generator) -> np.ndarray:
    e = np.zeros(ELEMENT_DIM)
    e[I_L] = L
    e[I_VRF] = np.exp(rng.uniform(np.log(VRF_RANGE[0]), np.log(VRF_RANGE[1])))
    e[I_FRF] = np.exp(rng.uniform(np.log(FRF_RANGE[0]), np.log(FRF_RANGE[1])))
    e[I_PHIRF] = rng.uniform(*PHIRF_RANGE)
    return e


# ==========================================================================
# Element constructors (randomly sampled parameters) — used by legacy modes
# ==========================================================================

def _sample_drift(rng: np.random.Generator) -> np.ndarray:
    return _make_drift(rng.uniform(*L_DRIFT))


def _sample_quad(rng: np.random.Generator, sign: int = 0) -> np.ndarray:
    L = rng.uniform(*L_QUAD)
    k1l = rng.uniform(*K1L_RANGE)
    if sign == 0:
        sign = rng.choice([-1, 1])
    return _make_quad(L, sign * k1l / L)


def _sample_sextupole(rng: np.random.Generator) -> np.ndarray:
    L = rng.uniform(*L_SEXT)
    k2l = rng.uniform(*K2L_RANGE)
    return _make_sextupole(L, k2l / L * rng.choice([-1, 1]))


def _sample_dipole(rng: np.random.Generator) -> np.ndarray:
    return _make_bend(rng.uniform(*L_DIPOLE), rng.uniform(*ANGLE_RANGE))


def _sample_rf(rng: np.random.Generator) -> np.ndarray:
    return _make_rf(rng.uniform(*L_RF), rng)


# ==========================================================================
# Sectioned lattice
# ==========================================================================

def _fodo_k1(mu: float, L_q: float, L_d: float) -> float:
    """Compute quad K1 from phase advance, quad length, and drift length.

    Uses the thin-lens FODO relation:
        cos(mu) = 1 - L_half^2 / (2 * f^2)
    where L_half = L_d + L_q is the half-cell center-to-center distance
    and f = 1 / (K1 * L_q) is the focal length.

    Args:
        mu: Phase advance per cell (rad), must be in (0, pi).
        L_q: Quad length (m).
        L_d: Drift-equivalent length between quads (m).

    Returns:
        K1 in m^-2 (always positive; sign assigned by caller).
    """
    L_half = L_d + L_q
    # 1/f = sqrt(2 * (1 - cos(mu))) / L_half
    inv_f = np.sqrt(2.0 * (1.0 - np.cos(mu))) / L_half
    return inv_f / L_q


def fodo_periodic_twiss(mu: float, L_half: float) -> tuple[float, float]:
    """Compute periodic Twiss at the focusing quad center.

    Thin-lens FODO result:
        beta_max = L_half * (1 + sin(mu/2)) / sin(mu)
        beta_min = L_half * (1 - sin(mu/2)) / sin(mu)

    Args:
        mu: Phase advance per cell (rad).
        L_half: Half-cell center-to-center distance (m).

    Returns:
        (beta_x, beta_y) at the QF location.
        beta_x = beta_max (beam is wide in the focusing plane at QF),
        beta_y = beta_min (beam is narrow in the defocusing plane at QF).
    """
    sin_mu = np.sin(mu)
    sin_half = np.sin(mu / 2.0)
    beta_max = L_half * (1.0 + sin_half) / sin_mu
    beta_min = L_half * (1.0 - sin_half) / sin_mu
    return beta_max, beta_min


def _build_straight_half_cell(
    L_q: float, K1: float, quad_sign: int, L_d: float,
    p_rf: float, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build one straight half-cell: quad + drift (or RF).

    With probability p_rf, the drift is replaced by an RF cavity of the
    same length (transversely equivalent to a drift).
    """
    elements = [_make_quad(L_q, quad_sign * K1)]
    if rng.random() < p_rf:
        elements.append(_make_rf(L_d, rng))
    else:
        elements.append(_make_drift(L_d))
    return elements  # 2 elements


def _build_arc_half_cell(
    L_q: float, K1: float, quad_sign: int, L_d: float,
    base_angle: float, p_sext: float, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build one arc half-cell: quad + [sext] + drift + bend + drift.

    The drift space of length L_d is split into drift + bend + drift,
    preserving the total drift-equivalent length. Optionally a sextupole
    is placed near the quad (its sign matches the quad sign).

    Args:
        L_q: Quad length (m).
        K1: Quad strength (m^-2), positive. Sign applied via quad_sign.
        quad_sign: +1 for QF, -1 for QD.
        L_d: Total drift-equivalent length (m).
        base_angle: Bend angle for this section (rad).
        p_sext: Probability of inserting a sextupole.
        rng: Random generator.
    """
    elements = [_make_quad(L_q, quad_sign * K1)]

    # Sextupole near quad (sign matches quad for chromaticity correction)
    if rng.random() < p_sext:
        sext_L = rng.uniform(*L_SEXT)
        sext_L = min(sext_L, L_d * 0.25)  # don't eat too much drift
        k2l = rng.uniform(*K2L_RANGE)
        elements.append(_make_sextupole(sext_L, quad_sign * k2l / sext_L))
        L_remaining = L_d - sext_L
    else:
        L_remaining = L_d

    # Split remaining space: drift + bend + drift
    bend_frac = rng.uniform(0.3, 0.6)
    bend_L = L_remaining * bend_frac
    drift_total = L_remaining - bend_L
    split = rng.uniform(0.3, 0.7)
    d1 = drift_total * split
    d2 = drift_total - d1

    # Perturb bend angle slightly
    angle = base_angle * (1.0 + rng.normal(0, 0.1))
    angle = max(0.005, angle)

    elements.append(_make_drift(d1))
    elements.append(_make_bend(bend_L, angle))
    elements.append(_make_drift(d2))

    return elements  # 4 or 5 elements


def sample_sectioned_lattice(
    seq_len: int, rng: np.random.Generator, *, linear: bool = False,
    n_sections: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a lattice from stitched FODO-based sections.

    Each section is a stable FODO channel with its own phase advance,
    cell length, and element insertions. Sections can be:
      - straight: quads + drifts, optional RF cavities
      - arc: quads + drifts + dipoles, optional sextupoles

    The lattice always starts at a half-cell boundary (quad first).

    Args:
        seq_len: Target number of elements.
        rng: NumPy random generator.
        linear: If True, no sextupoles or RF cavities.
        n_sections: Number of sections. If None, sampled randomly from {2, 3}.
            Set to 1 for single-section lattices with no cross-section mismatch.

    Returns:
        elements: (seq_len, 7) element parameter array.
        lattice_info: Dictionary with section metadata and periodic Twiss
            of the first section (for initial beam matching).
    """
    # --- 1. Sample section layout ---
    if n_sections is None:
        n_sections = rng.integers(2, 4)  # 2 or 3
    section_types = rng.choice(["straight", "arc"], size=n_sections)

    # --- 2. Sample cell parameters per section ---
    sections = []
    for stype in section_types:
        mu = rng.uniform(*MU_RANGE)
        L_q = rng.uniform(*L_QUAD)
        L_d = rng.uniform(*L_HALF_CELL)
        K1 = _fodo_k1(mu, L_q, L_d)
        sigma_pert = rng.uniform(*SIGMA_PERT_RANGE)

        sec = {
            "type": str(stype),
            "mu": float(mu),
            "L_q": float(L_q),
            "L_d": float(L_d),
            "K1": float(K1),
            "sigma_pert": float(sigma_pert),
        }

        if stype == "straight":
            sec["p_rf"] = 0.0 if linear else float(rng.uniform(*P_RF_RANGE))
        else:
            sec["p_sext"] = 0.0 if linear else float(rng.uniform(*P_SEXT_RANGE))
            sec["base_angle"] = float(rng.uniform(*ANGLE_RANGE))

        sections.append(sec)

    # --- 3. Allocate half-cells per section ---
    # Cost per half-cell: straight = 2 elements, arc = 4 (average, ignoring sext)
    costs = [2 if s["type"] == "straight" else 4 for s in sections]

    # Start with minimum 2 half-cells (1 full FODO cell) per section
    half_cells = [2] * n_sections
    budget_used = sum(half_cells[i] * costs[i] for i in range(n_sections))

    # Distribute remaining budget randomly
    remaining = seq_len - budget_used
    while remaining > 0:
        eligible = [i for i in range(n_sections) if costs[i] <= remaining]
        if not eligible:
            break
        i = rng.choice(eligible)
        half_cells[i] += 1
        remaining -= costs[i]

    # --- 4. Build elements ---
    elements = []
    quad_sign = 1  # always start with QF

    for sec_idx, sec in enumerate(sections):
        for _ in range(half_cells[sec_idx]):
            if len(elements) >= seq_len:
                break

            # Per-cell perturbation
            K1_pert = sec["K1"] * (1.0 + rng.normal(0, sec["sigma_pert"]))
            K1_pert = max(K1_pert, 0.1)  # keep focusing positive
            L_d_pert = sec["L_d"] * (1.0 + rng.normal(0, sec["sigma_pert"] * 0.5))
            L_d_pert = max(0.2, L_d_pert)

            if sec["type"] == "straight":
                hc = _build_straight_half_cell(
                    sec["L_q"], K1_pert, quad_sign, L_d_pert,
                    sec["p_rf"], rng,
                )
            else:
                hc = _build_arc_half_cell(
                    sec["L_q"], K1_pert, quad_sign, L_d_pert,
                    sec["base_angle"], sec["p_sext"], rng,
                )

            elements.extend(hc)
            quad_sign *= -1

    # Truncate to exactly seq_len (may cut last half-cell short)
    elements = elements[:seq_len]

    # Pad with drifts if under budget (rare, from rounding)
    while len(elements) < seq_len:
        elements.append(_make_drift(rng.uniform(0.5, 1.5)))

    # --- 5. Compute first section's periodic Twiss for beam matching ---
    first = sections[0]
    L_half = first["L_d"] + first["L_q"]
    beta_x_ref, beta_y_ref = fodo_periodic_twiss(first["mu"], L_half)

    elements_arr = np.array(elements)
    rf_freqs = elements_arr[elements_arr[:, I_FRF] > 0, I_FRF]

    lattice_info = {
        "sections": sections,
        "half_cells_per_section": [int(h) for h in half_cells],
        "beta_x_periodic": float(beta_x_ref),
        "beta_y_periodic": float(beta_y_ref),
        "mu_first_section": float(first["mu"]),
        "max_f_rf_GHz": float(rf_freqs.max()) if len(rf_freqs) > 0 else 0.0,
    }

    return elements_arr, lattice_info


# ==========================================================================
# Legacy lattice modes (kept for backward compatibility)
# ==========================================================================

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

    elements = elements[:seq_len]
    return np.array(elements)


def sample_fodo_lattice(seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an alternating-quad lattice with only quads and drifts.

    Layout: Q_F - drift - Q_D - drift - Q_F - drift - ...
    Same parameter ranges as structured mode but no dipoles, RF, or sextupoles.

    Args:
        seq_len: Number of elements in the lattice.
        rng: NumPy random generator.

    Returns:
        (seq_len, 7) array of element parameters.
    """
    elements = []
    quad_sign = rng.choice([-1, 1])

    while len(elements) < seq_len:
        elements.append(_sample_quad(rng, sign=quad_sign))
        quad_sign *= -1
        if len(elements) >= seq_len:
            break
        elements.append(_sample_drift(rng))

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


# ==========================================================================
# Bmad file writer
# ==========================================================================

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
