# Experiment Log

**Fixed across all runs:** `data/encoded_sectioned_10k` · `fusion=concat` · `lr=3e-4` · `batch_size=128` · `latent_dim=256` · `n_heads=8` · `mlp_ratio=4`

---

## Scan 1 — d_model sweep (2026-04-06)

**Question:** How does model width affect performance? Are 200 epochs sufficient?

### 1a. Short run (200 epochs, cosine, `scan1_dmodel`)

Fixed: `n_layers=6`, `dropout=0.1`, `wd=0.01`

| d_model | best val_loss |
|---------|--------------|
| 32      | 0.061        |
| 64      | 0.055        |
| 128     | 0.046        |
| 512     | 0.028        |

### 1b. Long run (500 epochs, cosine, `scan1_dmodel_long`)

| d_model | best val_loss |
|---------|--------------|
| 64      | 0.051        |
| 128     | 0.033        |
| 256     | 0.031        |
| 512     | 0.026        |

### 1c. Scheduler comparison (500 epochs, `scan1_rop`)

Same as 1b but with ReduceOnPlateau instead of cosine.

| d_model | cosine | ROP   |
|---------|--------|-------|
| 64      | 0.051  | 0.054 |
| 128     | 0.033  | 0.034 |
| 256     | 0.031  | 0.032 |
| 512     | 0.026  | 0.027 |

**Conclusions:**
- Larger d_model monotonically improves performance throughout the range tested.
- 200 epochs under-converges; 500 epochs gives meaningfully better results.
- Cosine scheduler consistently beats ReduceOnPlateau by a small but consistent margin.
- Best config from this scan: `d_model=512`, cosine, 500 epochs → val_loss **0.026**.

---

## Scan 2 — n_layers sweep (2026-04-06)

**Question:** How many transformer layers are needed?

Fixed: `dropout=0.1`, `wd=0.01`, 500 epochs, cosine scheduler.
L=6 baseline taken from scan1_dmodel_long.

### d_model=256

| n_layers | best val_loss |
|----------|--------------|
| 1        | 0.062        |
| 2        | 0.045        |
| 3        | 0.035        |
| 4        | 0.035        |
| 6        | 0.031        |

### d_model=512

Runs completed 2026-04-13 (SLURM job 51517619 cancelled; rerun locally on 4 GPUs in parallel).

| n_layers | best val_loss |
|----------|--------------|
| 1        | 0.055        |
| 2        | 0.038        |
| 3        | 0.030        |
| 4        | 0.027        |
| 6        | 0.026 (baseline from scan1) |

### Combined comparison

| n_layers | d_model=256 | d_model=512 |
|----------|-------------|-------------|
| 1        | 0.062       | 0.055       |
| 2        | 0.045       | 0.038       |
| 3        | 0.035       | 0.030       |
| 4        | 0.035       | 0.027       |
| 6        | 0.031       | 0.026       |

**Conclusions:**

The depth-scaling behaviour differs meaningfully between the two widths. At d_model=256, performance saturates between L=3 and L=4 (both 0.035), with only a small additional gain at L=6 (0.031). At d_model=512, improvement continues strongly through L=4 (0.027 vs 0.030 at L=3), with L=4 nearly matching L=6 (0.026). This suggests that wider models extract more signal from additional depth before saturating, but that saturation still occurs — the gap between L=4 and L=6 at d512 (0.001) is much smaller than the gap between L=3 and L=4 (0.003).

Across both widths, d_model=512 consistently outperforms d_model=256 at every layer count, and the gains are largest at intermediate depth (L=2–4), where the wider model has more capacity to exploit the representational headroom.

L=6 remains the best tested depth at both widths. Whether going deeper (L=8+) would yield further gains at d512 is an open question, but the diminishing returns from L=4→L=6 suggest the model is approaching saturation. The current best config (d512, L6) is likely close to the optimal depth–width trade-off within the tested range.

---

## Scan 3 — Regularization sweep (2026-04-06)

**Question:** Is the model overfitting? Does more dropout or weight decay help?

Fixed: `d_model=512`, `n_layers=6`, 500 epochs, cosine scheduler.
Baseline (dropout=0.1, wd=0.01) val_loss = **0.026**.

| dropout | weight_decay | best val_loss |
|---------|-------------|--------------|
| 0.1     | 0.01        | 0.026 (baseline) |
| 0.2     | 0.01        | 0.028        |
| 0.3     | 0.01        | 0.028        |
| 0.2     | 0.05        | 0.027        |
| 0.3     | 0.05        | 0.028        |

**Conclusions:**
- All regularized variants are slightly worse than the baseline.
- The model is not overfitting at this scale — extra regularization hurts optimization without providing a generalization benefit.
- Stick with `dropout=0.1`, `wd=0.01`.

---

## Evaluation — Best Cumsum Checkpoint (2026-04-13)

Checkpoint: `lbd_d512_L6_260406_185838_best.pth` (epoch 456, val_loss 0.0263, LatticeTransformer).
Output: `runs/lbd_d512_L6_260406_185838/eval/`.

### Per-step MSE

Error grows monotonically along the sequence — from ~2×10⁻⁴ at element 0 to ~8×10⁻² at element 31, roughly a 400× increase. Despite LatticeTransformer being a parallel (non-autoregressive) model, the cumsum output structure causes errors in early Δz predictions to propagate forward through the accumulated trajectory.

### Physical-space plots (VAE-decoded)

**Phase space portraits** (sample 317, 63rd percentile): qualitatively excellent. The model correctly captures the rotation of transverse phase ellipses (x-x', y-y') as the beam advances, and gets the round longitudinal distribution (z-δ) right throughout the sequence.

**Scale errors** (relative, pred−gt)/gt:
- σ_δ (energy spread): best-predicted dimension, errors within ±4% even for hard samples. Energy spread is approximately conserved through the lattice.
- σ_z (bunch length): worst-predicted dimension, up to 80-100% relative error on hard samples. RF cavities drive large bunch length changes that are the main source of difficulty.
- Transverse scales (σ_x, σ_x', σ_y, σ_y'): intermediate, ±20-40%.

**Centroid errors** (absolute, |pred−gt|):
- Transverse centroids (x, y, x', y') and energy centroid (δ) are very well predicted across all samples.
- Longitudinal centroid (z) for the 90th-percentile sample reaches ~0.6 units — the single largest outlier.

### Key takeaways

The model is qualitatively correct and quantitatively accurate on most phase-space dimensions. The systematic weakness is longitudinal dynamics (σ_z, z centroid), driven by RF elements. This is the clearest direction for future improvement — whether through architecture changes, loss reweighting toward later elements or the longitudinal dimensions, or more diverse training data covering a wider range of RF configurations.

---

## Scan 4 — Direct prediction output mode (2026-04-14)

**Question:** Does replacing the cumsum output with direct per-position displacement prediction fix the error accumulation problem?

**Motivation:** Evaluation of the best cumsum checkpoint (`lbd_d512_L6_260406_185838`) revealed that per-step latent MSE grows ~400× from element 0 to element 31. The root cause is structural: the LatticeTransformer predicts per-element increments Δzᵢ and recovers the trajectory via cumsum, but it only sees z₀ (via AdaLN conditioning) — never the intermediate states. Any systematic per-step bias accumulates over the sequence. By element 32, the error is the sum of 32 independent error contributions.

The fix introduces `output_mode=direct` in `LatticeConfig`. Instead of `z_t = z₀ + Σ Δzᵢ`, the model predicts the total displacement from z₀ directly at each position: `z_t = z₀ + disp_t`. With causal attention and AdaLN conditioning on z₀, each token has all the information needed to predict the total displacement to that point. Crucially, errors at each position are now independent — a bad prediction at element 5 does not contaminate elements 6–32.

**What we expect:** The per-step MSE curve should flatten substantially or grow much more slowly. Val loss may improve or stay similar — the flat MSE loss already penalizes late-position errors, so the training objective isn't the bottleneck. The benefit should show most clearly in evaluation (the per-step MSE plot), not necessarily in the scalar val loss.

**What the result implies either way:** If error accumulation disappears, the cumsum was the structural bottleneck and direct prediction is the right output architecture going forward. If the MSE curve still grows steeply, the problem lies elsewhere — possibly that predicting total displacement over 32 elements in one shot is genuinely harder and the model underfits at later positions, or that the loss weighting needs to change.

**Design:** d_model scan at {64, 128, 256, 512}, n_layers=6, 500 epochs, all other hyperparameters at best-config values. Mirrors Scan 1b so results are directly comparable across both output modes.

| d_model | best val_loss (direct) | best val_loss (cumsum, scan 1b) |
|---------|------------------------|----------------------------------|
| 64      | 0.026                  | 0.051                            |
| 128     | **0.023**              | 0.033                            |
| 256     | 0.024                  | 0.031                            |
| 512     | 0.025                  | 0.026                            |

Runs: `scan_direct_dmodel_d{64,128,256,512}_260414_1213`.

**Results and analysis:**

Direct prediction is a clear win at small-to-medium model sizes and produces the new overall best checkpoint. At d_model=128, direct mode achieves val_loss=0.023, a 30% improvement over the cumsum baseline at the same width (0.033) and a 12% improvement over the best cumsum result at any width (d512=0.026). The gains are largest where the cumsum model struggled most: at d64, direct mode halves the val_loss (0.026 vs 0.051), and at d128 the improvement is also large (0.023 vs 0.033).

At d_model=512, however, direct mode fails to improve on cumsum and is actually marginally worse (0.025 vs 0.026). The explanation is in the train/val gap: train_loss at d512 is 0.00049 — two orders of magnitude below val_loss (0.025) — indicating severe overfitting. Direct prediction is a structurally harder task than cumsum: instead of predicting a small local increment, the model must predict the total accumulated displacement to each position. This requires stronger generalization. Larger models, with more capacity to memorize training trajectories, overfit more aggressively under this harder objective. The training loss gap grows monotonically with model size (0.008→0.003→0.001→0.0005), while val_loss has a non-monotone sweet spot at d128.

This shifts the optimal width from d512 (under cumsum) to d128 (under direct). The d128 direct model is 15× smaller than d512 cumsum while achieving better generalization — a significant efficiency gain. Whether deeper networks at d128 (L=8, L=12) would recover some of the lost capacity without overfitting is worth investigating.

The key architectural conclusion is that direct prediction is the better output mode: it produces lower val_loss across most of the width range and only breaks down for over-parameterized models. Going forward, direct mode should be the default. The next step is either a regularization scan at d512 with direct mode (to recover that regime) or a depth scan at d128 with direct mode (to push the best result lower).

---

## Dataset — Single-section lattices (2026-04-14)

**Motivation:** All training to date used `encoded_sectioned_10k`, generated with the multi-section lattice sampler (2–3 FODO sections per lattice). Evaluation of the best checkpoint revealed a monotonically growing per-step MSE — roughly 400× higher at element 31 than at element 0. While Scan 4 is investigating whether the cumsum output structure is the architectural cause, there is a separate data-side explanation worth ruling out: the training distribution has a systematic bias where element index correlates with dynamics complexity.

The bias arises from how beam matching works. `sample_matched_beam_params` matches the initial beam to Section 1's periodic Twiss (with a B_mag mismatch factor of 1–5). At the Section 1→2 boundary, the beam carries whatever Twiss state Section 1 dynamics left it in, which is generally mismatched to Section 2's periodic solution by an uncontrolled amount. This produces compounding envelope beating in later sections. The result is that every training sample has "easy" dynamics near element 0 and "hard" dynamics near element 31 — always in that order. A model trained on this distribution learns the positional difficulty gradient as a fact about the world rather than an artifact of the dataset.

**Fix:** Generate single-section lattices (`--n-sections 1`). With a single FODO section, the beam is matched to the section's optics throughout the sequence with only the controlled B_mag mismatch. There are no cross-section transitions, so difficulty is distributed uniformly across sequence positions. Diversity is preserved through μ ∈ [20°–80°], cell perturbation σ ∈ [0–25%], B_mag ∈ [1–5], element insertions (RF, sextupoles, dipoles), emittance, and energy spread.

**Dataset:** `encoded_sectioned_1sec_10k` — 10,000 samples, seq_len=32, n_sections=1, seed=200. Same size as the existing dataset to isolate the effect of the lattice structure change.

**What we expect:** Training on this dataset should produce a flatter per-step MSE curve. If the positional bias was the dominant cause of the ~400× MSE growth, the new model should show near-uniform error across sequence positions. If the MSE curve still grows steeply (even on clean data), the architectural explanation from Scan 4 — cumsum error accumulation — is the real bottleneck and fixing the data alone is insufficient.

**What the result implies either way:** If flatter MSE and better val loss: the multi-section dataset was the primary problem, and all future datasets should use single-section generation. If MSE still grows: the dataset bias was a secondary effect and the architectural fix (direct output mode, Scan 4) is the critical intervention. Either way, the result cleanly separates data quality from architecture as the source of error accumulation.

---

## Current Best Config (as of 2026-04-14)

| Hyperparameter  | Value                                    |
|-----------------|------------------------------------------|
| output_mode     | direct                                   |
| d_model         | 128                                      |
| n_layers        | 6                                        |
| n_heads         | 8                                        |
| mlp_ratio       | 4                                        |
| dropout         | 0.1                                      |
| weight_decay    | 0.01                                     |
| lr              | 3e-4                                     |
| scheduler       | cosine                                   |
| epochs          | 500                                      |
| batch_size      | 128                                      |

**val_loss = 0.0232** · run `scan_direct_dmodel_d128_260414_1213`

---

## Open Questions

- **Depth scan at d128 with direct mode.** The d128 direct model is the current best, and it is not obviously saturating on depth — a scan over L=4,6,8,12 could push the val_loss lower without the overfitting risk of larger widths.
- **Regularization for d512 direct.** Direct mode at d512 severely overfits (train_loss=0.0005 vs val_loss=0.025). Stronger regularization (higher dropout, weight decay, or data augmentation) may recover the large-capacity regime.
- **Can longitudinal prediction be improved?** The model's main failure mode is σ_z and z centroid errors driven by RF elements. Possible approaches: loss reweighting by element type or sequence position, explicit RF element conditioning, or longer training sequences.
- **Single-section dataset effect.** `encoded_sectioned_1sec_10k` is generated and encoded but not yet used for training. Results pending.
