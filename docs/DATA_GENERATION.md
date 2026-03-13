# Data Generation Guide

This document describes how to generate training data for the latent beam dynamics
model. It covers parameter selection for accelerator lattice elements and initial
beam distributions, with physics reasoning for every choice.

## Overview

The training data consists of open (non-periodic) lattice segments — transfer lines,
linac sections, or arbitrary element sequences — through which a particle beam is
tracked. Each training sample is one lattice + one initial beam tracked through it.

### Pipeline

```
Sample lattice  -->  Sample initial beam  -->  Track (Bmad)  -->  Encode (VAE)  -->  Save .npy
```

### Output format

| File | Shape | Description |
|------|-------|-------------|
| `z_traj.npy` | `(N, seq_len+1, latent_dim)` | VAE-encoded beam state at each element boundary |
| `elements.npy` | `(N, seq_len, element_dim)` | Element parameters per element |

The 7 element parameters per element are `[L, K1, K2, Angle, V_rf, f_rf, phi_rf]`,
matching the model's `ElementEncoder` input.


---

## 1. Element Parameter Ranges

### 1.1 Quadrupole gradient K1: +/-0.5 to 5 m^-2

**What it is.** K1 = eG/(p/c), the normalized quadrupole gradient, where G = dB/dx
is the magnetic field gradient and p is the beam momentum.

**Upper bound (5 m^-2) — magnet technology and optical stability.**

The achievable gradient is set by magnet technology. Normal-conducting quadrupoles
produce G = 5-20 T/m. Superconducting quads reach 100+ T/m but are uncommon in
general lattice design.

For a beam at energy E with rigidity B*rho = p/e:

| Energy | B*rho (T*m) | K1 at G=20 T/m |
|--------|-------------|-----------------|
| 0.5 GeV | 1.67 | 12 m^-2 |
| 1 GeV | 3.34 | 6 m^-2 |
| 3 GeV | 10.0 | 2 m^-2 |
| 10 GeV | 33.4 | 0.6 m^-2 |

So K1 = 0.5-5 covers the range for ~0.5-10 GeV beams with conventional magnets.

From the optics side: the focal length of a quadrupole is f = 1/(K1*L_quad). With
K1 = 5 and L_quad = 0.3 m, f = 0.67 m. When f becomes shorter than the spacing
between elements, the phase advance per element exceeds pi and the beam dynamics
become degenerate — essentially exponential blowup within a few elements. K1 > 10
with typical element lengths puts you firmly in this regime, producing training data
that is just exponential growth and not informative.

**Lower bound (0.5 m^-2) — optical relevance.**

With K1 = 0.5 and L_quad = 0.3 m, f = 6.7 m. For drift spaces of 1-5 m, this quad
barely deflects the beam — it's effectively a drift. Below K1 = 0.5, the quadrupole
effect is negligible and the element is not contributing interesting dynamics.

**Sampling.** Uniform in [-5, -0.5] union [0.5, 5], or equivalently: sample the sign,
then sample |K1| uniformly in [0.5, 5]. Alternating signs between successive quads
produces FODO-like focusing that keeps beams reasonably contained.


### 1.2 Sextupole gradient K2: +/-1 to 10 m^-3

**What it is.** K2 = (1/B*rho) * d^2B/dx^2, the normalized sextupole gradient.
Sextupoles provide nonlinear kicks used for chromaticity correction.

**Sextupoles must be perturbative relative to quadrupoles.** The sextupole kick at
transverse displacement x is proportional to K2*x^2/2, versus the quadrupole kick
K1*x. Their ratio is:

    K2*x / (2*K1)

For this to remain below ~10% at a typical beam displacement of x = 5 mm:

    K2 * 0.005 / (2 * K1) < 0.1
    K2 < 40 * K1

With K1 in [0.5, 5], this gives K2 < 20-200. The range 1-10 is well within the
perturbative regime and matches chromaticity correction strengths in real machines.
Going far above this makes the sextupole dominate over the quadrupole, producing
dynamics that are unphysically nonlinear.

**Sampling.** Uniform in [-10, 10]. Many elements will have K2 = 0 (only elements
designated as sextupoles get nonzero K2).


### 1.3 Element lengths

**Drifts: 0.2 to 5 m.**

- Lower bound (0.2 m): Below this, there is not enough physical space for hardware
  between magnets (BPMs, vacuum pumps, bellows, correctors). Drift spaces shorter
  than ~20 cm are rare in real lattices.
- Upper bound (5 m): Long drifts produce trivial free-space beam expansion
  (sigma grows as sqrt(1 + (s/beta)^2)). Beyond 5 m the dynamics are simple and
  don't add much to the training set. Can extend to 10 m if desired.

**Quadrupoles: 0.1 to 0.5 m.**

- Lower bound (0.1 m): Approaching the thin-lens limit. Hard to fabricate with good
  field quality. Also, K1*L must be large enough for the quad to have an optical
  effect — very short quads need unrealistically high K1.
- Upper bound (0.5 m): Possible but uncommon. Most real quads are 0.15-0.4 m. APS
  quads are 0.2-0.5 m, LCLS quads 0.1-0.3 m.

**Dipoles: 0.5 to 3 m.**

- Lower bound (0.5 m): Very short for a bending magnet. Below this, the bend angle
  per element is tiny.
- Upper bound (3 m): Most dipoles are 1-3 m. Some ring dipoles are longer (APS:
  6.6 m) but 3 m gives sufficient variation for open lattices. Can extend to 5 m.

**RF cavities: 0.1 to 2 m.**

- Normal-conducting cavities are typically 0.3-2 m. Superconducting cavities (e.g.,
  1.04 m for TESLA-type nine-cell) also fall in this range.

**Sampling.** Uniform or log-uniform within each element type's range. The type of
each element determines which length range to draw from.


### 1.4 Dipole bend angle: 0 to 0.5 rad

In a storage ring with N dipoles, the angle per dipole is 2*pi/N. With 30-60 dipoles
this gives 0.1-0.2 rad per dipole. Transfer line bends are typically smaller,
0.01-0.1 rad. 0.5 rad (about 29 degrees) is already a large single-element bend.

Above 0.5 rad per element is physically unusual and creates very strong edge-focusing
and dispersion effects that dominate the dynamics in unrealistic ways.

**Sampling.** Uniform in [0.01, 0.5] for dipole elements. Non-dipoles have Angle = 0.


### 1.5 RF voltage V_rf: 0.1 to 10 MV

Normal-conducting cavities have gradients of 1-5 MV/m with lengths of 0.5-2 m,
giving 0.5-10 MV per cavity. Superconducting cavities reach 15-25 MV/m but are
specialized structures. 0.1 MV is the lower bound below which the RF effect on the
beam is negligible.

**Sampling.** Log-uniform in [0.1, 10] MV for RF elements. Non-RF elements have
V_rf = 0.


### 1.6 RF frequency f_rf: 0.1 to 3 GHz

This range covers essentially all real accelerator RF systems:
- 352 MHz (ESS)
- 500 MHz (APS, ALS)
- 650 MHz (PIP-II)
- 1.3 GHz (LCLS-II, ILC)
- 2.856 GHz (SLAC S-band)

Below 100 MHz: RF structures become impractically large. Above 3 GHz: X-band
territory, uncommon. For a general-purpose model, 0.1-3 GHz spans the field.

**Sampling.** Log-uniform in [0.1, 3] GHz for RF elements. Log-uniform is appropriate
because RF frequencies cluster at specific bands spanning an order of magnitude.


### 1.7 RF phase phi_rf: -pi/6 to pi/6

- phi = 0 (on-crest): maximum energy gain.
- |phi| < pi/6 (~30 degrees): normal operating range. Provides acceleration with
  controllable energy chirp for bunch compression.
- |phi| > pi/3: weak or zero net acceleration. Approaching pi gives deceleration.

The range -pi/6 to pi/6 covers typical operating conditions. Broaden to +/-pi/3 if
you want to include strong off-crest operation (e.g., aggressive bunch compression
in FEL injectors).

**Sampling.** Uniform in [-pi/6, pi/6] for RF elements.


### 1.8 Summary table

| Parameter | Range | Sampling | Reasoning |
|-----------|-------|----------|-----------|
| K1 | +/-[0.5, 5] m^-2 | Uniform, random sign | Magnet technology + optical stability |
| K2 | +/-[1, 10] m^-3 | Uniform | Perturbative relative to K1 |
| L (drift) | [0.2, 5] m | Uniform | Physical space limits |
| L (quad) | [0.1, 0.5] m | Uniform | Fabrication + field quality |
| L (dipole) | [0.5, 3] m | Uniform | Typical bending magnets |
| L (RF) | [0.1, 2] m | Uniform | Cavity dimensions |
| Angle | [0.01, 0.5] rad | Uniform | Edge effects dominate above 0.5 |
| V_rf | [0.1, 10] MV | Log-uniform | Spans normal/superconducting |
| f_rf | [0.1, 3] GHz | Log-uniform | Covers all standard RF bands |
| phi_rf | [-pi/6, pi/6] | Uniform | Typical on/near-crest operation |


---

## 2. Initial Beam Parameter Ranges

The initial beam is a 6D phase-space distribution characterized by Twiss parameters,
emittances, longitudinal parameters, and centroid offsets. Since these are open
lattices (not periodic), there is no matched condition — the initial beam is sampled
independently from the lattice.

### 2.1 Twiss beta: 0.5 to 100 m (log-sampled)

Beta determines beam size: sigma = sqrt(emittance * beta).

- Lower bound (0.5 m): A tight focus. For emittance = 1 nm*rad, sigma = 22 um.
  Below 0.5 m you're approaching collider interaction-point optics, which is a very
  specialized regime.
- Upper bound (100 m): A large, divergent beam. For emittance = 1 nm*rad,
  sigma = 0.3 mm. In well-designed lattices beta rarely exceeds 30-50 m. Above 100 m
  signals badly mismatched optics.

**Log-sampled** because beta varies over two orders of magnitude and you want uniform
coverage of beam sizes. Uniform sampling would over-represent large-beta (big beam)
cases.

Both beta_x and beta_y are sampled independently — there is no reason for them to
be correlated.


### 2.2 Twiss alpha: -3 to 3

Alpha = -beta'/2 measures beam convergence (alpha > 0) or divergence (alpha < 0).

- alpha = 0: beam waist (neither converging nor diverging).
- |alpha| ~ 1: moderate convergence/divergence, typical mid-lattice.
- |alpha| > 3: strongly converging/diverging, typically only near a very tight
  focus where beta is small and changing rapidly.

**Uniformly sampled** (not log) since alpha is naturally centered around zero and can
be negative.


### 2.3 Geometric emittance: 0.1 nm*rad to 10 um*rad (log-sampled)

Emittance measures the phase-space area occupied by the beam. This range spans the
full spectrum of real accelerators:

| Machine type | Typical emittance |
|--------------|-------------------|
| Electron storage rings (after damping) | 0.01-10 nm*rad |
| Electron guns (geometric, at 1 GeV) | 0.05-0.5 nm*rad |
| FEL injectors | 0.1-1 nm*rad |
| Proton machines | 1-10 um*rad |

- Lower bound (0.1 nm*rad): Below this, quantum effects (intrabeam scattering,
  quantum excitation) become significant and classical tracking is not fully valid.
- Upper bound (10 um*rad): Covers proton machines. Above this, beams are very large
  and likely to exceed physical apertures in any real system.

**Log-sampled** because emittance spans 5 orders of magnitude. Uniform sampling would
make everything look like a proton beam.

Both emittance_x and emittance_y are sampled independently. In real machines they
are often very different (e.g., electron rings have emittance_y << emittance_x).


### 2.4 Energy spread sigma_delta: 10^-4 to 10^-2 (log-sampled)

sigma_delta = sigma_E / E, the relative energy spread.

- Lower bound (10^-4): Typical for storage rings and well-conditioned linacs.
  Below this, chromatic effects (K1 * sigma_delta coupling) are negligible and
  sextupoles have no observable effect — the dynamics become purely linear and
  less interesting.
- Upper bound (10^-2): Large energy spread, seen in RF photoinjectors or beams
  with strong energy chirp. Above this, particles are likely outside the momentum
  acceptance of most lattice elements.

**Log-sampled** to uniformly cover the two decades.


### 2.5 Bunch length sigma_z: 0.1 mm to 10 cm (log-sampled)

- Lower bound (0.1 mm): Short electron bunches in FELs (e.g., LCLS compressed
  bunch ~20 um). Going below 0.1 mm is valid but rare.
- Upper bound (10 cm): Long proton bunches.

Bunch length matters because it determines the RF curvature effect: particles at
different longitudinal positions see different RF phase, leading to correlated
energy spread growth proportional to k_rf * sigma_z.

**Log-sampled** over the three decades.


### 2.6 Centroid offsets: 0 to 3 sigma

Sample offsets in each phase-space coordinate (x, px, y, py, z, delta) as multiples
of the beam's RMS size in that coordinate:

    offset_x  = n * sqrt(emittance_x * beta_x)       where n ~ Normal(0, 1), clipped to [-3, 3]
    offset_px = n * sqrt(emittance_x / beta_x)
    (similarly for y, py, z, delta)

**Why in units of sigma, not absolute units.** An offset of 1 mm means very different
things for a 10 um beam versus a 1 mm beam. Sampling in units of sigma keeps offsets
physically meaningful regardless of the emittance and Twiss parameters.

**Why clipped at 3 sigma.** Real beams are typically within 1-2 sigma of the design
orbit due to injection errors and alignment tolerances. Beyond 3 sigma, nonlinear
fields from sextupoles and magnet fringe fields dominate, and the beam would
typically be considered mis-steered.


### 2.7 Summary table

| Parameter | Range | Sampling | Reasoning |
|-----------|-------|----------|-----------|
| beta_x, beta_y | [0.5, 100] m | Log-uniform | Spans tight focus to large beam |
| alpha_x, alpha_y | [-3, 3] | Uniform | Centered at zero (beam waist) |
| emittance_x, emittance_y | [0.1 nm, 10 um]*rad | Log-uniform | Spans electrons to protons |
| sigma_delta | [10^-4, 10^-2] | Log-uniform | Below: no chromatic effects; above: lost beam |
| sigma_z | [0.1 mm, 10 cm] | Log-uniform | Spans FEL bunches to proton bunches |
| Centroid offsets | [0, 3] sigma | Normal(0,1) clipped | Physically meaningful at any emittance |


---

## 3. Lattice Construction

### 3.1 Element type vocabulary

Each element in the lattice is one of five types. Non-applicable parameters are
set to zero:

| Type | Nonzero parameters |
|------|-------------------|
| Drift | L |
| Quadrupole | L, K1 |
| Sextupole | L, K2 |
| Dipole | L, Angle |
| RF cavity | L, V_rf, f_rf, phi_rf |

This means most entries in the element parameter vector are zero. The model learns
element identity implicitly from the sparsity pattern.


### 3.2 Why not purely random lattices?

If you sample element types independently (e.g., 40% drift, 30% quad, ...), most
lattices produce the same uninteresting dynamics. The problem is element **ordering**:
random type sequences frequently place multiple focusing quads in a row, or long
stretches with no focusing at all. In both cases the beam blows up within a few
elements and the remaining 90% of the sequence is just exponential growth. The model
sees the same "blowup" pattern over and over and rarely encounters well-controlled
transport, chromaticity effects, or subtle nonlinear dynamics.

You could recover useful lattices by generating randomly and filtering aggressively
(beam size cap), but you'd discard the vast majority of what you generate. Structured
generation is an efficiency optimization: it biases element ordering toward the
dynamically interesting region of lattice space without constraining the parameters.


### 3.3 Two structural rules

All structured lattice generation reduces to two rules:

1. **Alternate focusing and defocusing quads.** Successive quadrupoles should have
   opposite signs of K1. This is the minimum requirement for net focusing in both
   transverse planes. Without sign alternation, quads focus in x but defocus in y
   (or vice versa), and random sign sequences almost always defocus in at least
   one plane within a few elements.

2. **Interleave drifts between non-drift elements.** Real accelerators have physical
   space between magnets for diagnostics, vacuum hardware, and correctors. Drifts
   between active elements also give the beam space to evolve between kicks, which
   produces richer dynamics than back-to-back magnets.

Everything else is a special case of these two rules.


### 3.4 Structured generation (50% of data)

The generator works as follows:

1. Lay down an alternating quad backbone: `QF - QD - QF - QD - ...`
2. Insert drifts between every element.
3. Optionally insert dipoles, RF cavities, or sextupoles at random positions
   between the quads.
4. Randomize all parameters independently from the ranges in Section 1.

This is NOT a periodic lattice — every element gets independent random parameters.
The only structural constraint is the two rules above.

The named lattice types are just variations on step 3:

**FODO-like.** Only quads and drifts. The simplest case.

```
Q(K1=+2.3, L=0.20) - drift(L=1.7) - Q(K1=-3.1, L=0.35) - drift(L=0.8) - Q(K1=+1.9, L=0.15) - ...
```

Note: K1 magnitudes differ between quads, drift lengths vary, nothing repeats. The
only constraint is alternating K1 sign.

**Transport arc.** Insert dipoles between some quad-drift pairs.

```
dipole - drift - QF - drift - QD - drift - dipole - drift - QF - ...
```

This adds bending and dispersion to the dynamics.

**Linac-like.** Insert RF cavities between some quad-drift pairs.

```
RF - drift - QF - drift - RF - drift - QD - drift - ...
```

This adds acceleration and longitudinal dynamics.

These categories can be mixed — a lattice can contain both dipoles and RF cavities.
The categories are a guide for ensuring diversity, not rigid templates.


### 3.5 Fully random lattices (30% of data)

Drop both structural rules. Sample element type per slot independently:

- 40% drift, 30% quadrupole, 15% dipole, 10% sextupole, 5% RF cavity
- Sample each element's parameters independently from the ranges in Section 1

Most of these will produce beam blowup. Apply the beam size cap filter (Section 4.1)
and keep the survivors. Expect to discard 50-80% of generated lattices.

The surviving random lattices provide dynamics that the structured generator cannot
produce — unusual element orderings that happen to be stable, asymmetric focusing
patterns, and other edge cases that improve model generalization.


### 3.6 Challenging dynamics (20% of data)

Deliberately construct lattices near interesting dynamical regimes:

- **Near-unstable optics:** Use the structured generator but push quad strengths
  toward the upper range (K1 near 5) where beta functions grow rapidly between
  quads. The beam stays contained but is near the stability boundary.
- **Strong chromatic effects:** High K1 combined with large energy spread beams
  (sigma_delta near 10^-2). Chromaticity causes different-energy particles to
  experience different focusing, spreading the beam in phase space.
- **Nonlinear-dominated:** Sextupoles at higher strengths (K2 up to 20, pushing
  the perturbative boundary) placed near quads where dispersion is large.


### 3.7 Sequence length

Fixed `seq_len` per dataset (e.g., 50, 100, or 200 elements). Can vary across
datasets if training for length generalization. 50-100 is a reasonable starting
point — enough elements for interesting dynamics without excessive computational
cost.


---

## 4. Filtering

Since these are open lattices without apertures, particles are never lost.
However, some lattices produce dynamics that are not useful for training:

### 4.1 Beam size cap

Track the beam sigma matrix (linear transport, no particles needed) through the
lattice. Discard if the RMS beam size exceeds a threshold at any point:

```
sigma_x = sqrt(emittance * beta_x_transported) > threshold
```

**Recommended threshold: 100x the initial beam size.** Beyond this the dynamics are
purely exponential blowup. The model can learn "beam grows" from moderate growth;
it does not need to see sigma go from microns to meters.

This filter primarily rejects Tier 2 (random) lattices with unfortunate quad
arrangements. Tier 1 lattices will rarely be rejected.

### 4.2 Trivial dynamics filter

Optionally discard lattices where the beam changes by less than 1% (e.g., a
sequence of very weak elements that is effectively all drifts). These add bulk
to the dataset without teaching the model anything.

### 4.3 No survival filter needed

With no apertures, all particles survive by definition. Do NOT filter on particle
survival — it is not meaningful here.


---

## 5. Targeting a Specific Machine

The ranges above are for a **general-purpose model** spanning multiple accelerator
types. If targeting a specific machine:

1. Look up the actual element parameters from the machine lattice file.
2. Narrow all ranges to +/-20-50% around the nominal values.
3. Sample beam parameters from the machine's design beam + perturbations.

This will produce a more focused dataset that trains faster and achieves higher
accuracy for that machine, at the cost of generalization.


---

## 6. Dataset Sizes

| Phase | N (lattice-beam pairs) | Purpose |
|-------|------------------------|---------|
| Debug | 1,000 | Overfit sanity check |
| Development | 10,000-50,000 | Hyperparameter search |
| Production | 100,000-500,000 | Final training |

Each sample is one lattice + one initial beam. The same lattice can be paired with
multiple beams (and vice versa) to decouple lattice diversity from beam diversity.
