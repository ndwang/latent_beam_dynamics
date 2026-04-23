# Experiment Log

**Defaults unless noted:** `fusion=concat` · `lr=3e-4` · `latent_dim=256` · `n_heads=8` · `mlp_ratio=4` · `dropout=0.1` · `wd=0.01` · `scheduler=cosine` · `500 epochs`

**Note on batch_size:** Scan 1 used `batch_size=128`; Scans 2–5 used `batch_size=32` (yaml default, no override). The header originally claimed 128 for all runs — this was wrong. Check `config.yaml` in each run directory for the true value.

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

**Training:** d128, L6, direct mode, 500 epochs (job 51568722). Run: `lbd_d128_L6_260414_201843`.

**Results and analysis:**

Best val_loss: **0.02195** at epoch 389. Train loss: 0.00199. A 5.4% improvement over the multi-section d128/L6/direct baseline (0.0232), confirming that the cleaner dataset helps marginally.

The per-step MSE curve is still monotonically growing — from ~3×10⁻⁴ at element 0 to ~5×10⁻² at element 31, a ~150× increase. The shape is not flat. This is a decisive result: the data-side hypothesis was wrong, or at least insufficient. Eliminating cross-section transitions reduced the growth ratio from ~400× to ~150× — a real improvement — but the qualitative pattern is unchanged. The model still learns to predict early elements well and fails increasingly at later ones.

This means the primary driver of positional MSE growth is architectural, not a dataset artifact. The LatticeTransformer conditions all predictions via AdaLN on z₀ alone. As the beam evolves through 32 elements, the current beam state diverges from z₀ and that single conditioning signal becomes a progressively worse proxy for where the beam actually is. The model has no mechanism to track this evolution — it must predict the total displacement from z₀ at element 30 using exactly the same conditioning information it used at element 1. The ~150× remaining growth is an irreducible consequence of this design given finite model capacity.

The steep initial rise in the first ~10 elements followed by a slower plateau (roughly log-linear) is consistent with this interpretation: the first few elements produce large Twiss-dependent variations that z₀ mostly captures, but which small errors in the initial prediction compound into a regime the model can't recover from.

**Physical-space failures (sample 967, 63rd percentile):**
- Transverse scales (σ_x, σ_x', σ_y, σ_y'): tracked with 10–40% relative error, missing fine oscillatory structure.
- **⟨z⟩ centroid: complete failure.** Ground truth shows large monotonic longitudinal drift driven by RF; prediction is essentially flat at zero. The model has no concept of accumulated RF phase kick.
- σ_z and σ_δ are reasonably well predicted.
- Phase space portraits (x-x', y-y', z-δ) are qualitatively correct.

**Conclusion:** The single-section dataset is strictly better than multi-section and should be used going forward — it removes an unnecessary confound and modestly improves val loss. But fixing the data does not fix the MSE growth. The architectural limitation (z₀-only conditioning) is the dominant bottleneck. Meaningful further improvement requires giving the model a way to condition on something closer to the current beam state, not just the initial one. The most direct path is prepending z₀ as a learned attention token so every element can attend to it directly, or exploring a recurrent/autoregressive conditioning mechanism.

---

## Scan 5 — Depth scan at d128, direct mode (2026-04-14)

**Question:** Does increasing depth beyond L=6 improve performance at d_model=128 with direct output mode, and where does it saturate?

**Motivation:** Scan 4 found that d128 with direct mode is the best configuration (val_loss=0.023), but all depth work in scans 2 was done at d256/d512 with cumsum. The depth–width interaction may differ under direct mode: d128 doesn't overfit, so adding layers increases capacity without the generalization penalty seen at d512. The question is whether the model is depth-saturated at L=6 or still has headroom.

**Design:** n_layers scan at {4, 6, 8, 12}, d_model=128, direct mode, 500 epochs. L=6 is a free repeat of the scan 4 d128 result.

| n_layers | best val_loss | train_loss |
|----------|--------------|------------|
| 4        | 0.02504      | 0.00463    |
| 6        | 0.02316      | 0.00301    |
| 8        | 0.02264      | 0.00253    |
| 12       | **0.02234**  | 0.00174    |

Runs: `scan_direct_depth_d128_L{4,6,8,12}_260414_1314`.

**Results and analysis:**

Depth helps consistently and without overfitting across the full range tested. Val_loss improves monotonically from L4 (0.025) through L12 (0.022), and the train/val gap remains modest at all depths — L12's train_loss of 0.002 vs val_loss of 0.022 is a 10× gap, in contrast to the 50× gap seen at d512 direct. The model is not capacity-limited at d128 even at L=12.

The returns diminish sharply after L=6. The L4→L6 gain is 0.002 — the only meaningful jump in the scan. Beyond that, L6→L8 is 0.0005 and L8→L12 is 0.0003, both well within noise and not worth the additional parameters. The L=6 result (0.023156) reproduces the scan 4 value exactly, confirming good run-to-run consistency.

L=6 is therefore the right operating point: it captures the only real depth gain in this scan and adding more layers buys nothing. Going deeper is not the path to improvement at this width — the flattening after L=6 suggests d128 is approaching its representational ceiling regardless of depth. Further gains likely require wider models or more data.

---

## Dataset — variety_1sec_100k (2026-04-14)

**Motivation:** All scans to date trained on `encoded_sectioned_10k` (10,000 multi-section samples) or the smaller `encoded_sectioned_1sec_10k` (10,000 single-section samples). Both are likely bottlenecked by dataset size — the latent space has 256 dimensions, the lattice parameter space is high-dimensional, and 10k samples is small. Scan 4 and 5 showed that the model still has room to improve, and the depth scan plateau at d128/L6 suggests the model may be data-limited rather than capacity-limited.

The new dataset addresses two things simultaneously: scale (10× more samples) and diversity (wider parameter ranges). The diversity expansion targets the known weak points. RF voltage is widened from 0.1–5 MV to 0.01–20 MV to cover a broader range of longitudinal dynamics, which is the model's current worst failure mode. Phase advance is extended from 20°–80° to 10°–85° to cover weak-focusing and strongly-focused regimes. Half-cell length is widened from 0.75–2.5 m to 0.5–5.0 m. RF probability per cell is increased from 0–30% to 0–60% so the model trains on more RF-dense lattices. Cell-to-cell perturbation is increased to 0–40%. On the beam side, B_mag is extended from 1–5 to 1–10, energy from 0.5–10 GeV to 0.1–20 GeV, and energy spread from 1e-4–5e-3 to 1e-4–1e-2. The RF phase range is kept at ±30° (appropriate for storage ring RF operating near the stable fixed point). All other conventions are unchanged: single-section, seq_len=32, seed=300.

The generation pipeline was also improved: generation is now distributed across all 4 allocated nodes (512 CPUs total) rather than running on the head node only, cutting generation time from ~66 min to ~17 min for 100k samples. Reproducibility is preserved — seeds are derived from the full `SeedSequence.spawn(n_samples)` on every node, so sample `i` always gets the same seed regardless of node count.

**Dataset:** `encoded_variety_1sec_100k` — 100,000 samples, seq_len=32, n_sections=1, seed=300.

**What we expect:** A significant val_loss reduction relative to training on 10k samples, driven by both scale and diversity. The longitudinal prediction failures (σ_z, z centroid) should improve specifically from the wider RF voltage range and higher RF density. Whether the model generalizes better to unseen lattice configurations will show in the gap between the best 10k val_loss (0.022) and the new result.

**Jobs:** Generation completed (51568753). Encoding job (51620879) was cancelled — dataset deleted before encoding completed.

**Outcome:** Dataset deleted due to storage quota. The raw tracking output is ~200 MB per sample (33 beam snapshots × 100k particles × 6D × float64), so 100k samples accumulates ~20 TB before encoding. This exceeds the available scratch quota. The generate-then-encode pipeline does not scale beyond ~10k samples without a redesigned approach that encodes and discards raw particle data incrementally rather than accumulating the full dataset first.

---

## Architecture Comparison — TrackingTransformer and DualStreamTransformer (2026-04-16)

**Question:** Do TrackingTransformer or DualStreamTransformer outperform LatticeTransformer? The LatticeTransformer conditions all predictions via AdaLN on z₀ alone, which was identified as the dominant architectural bottleneck (per-step MSE grows ~150× across the sequence even with direct output mode and single-section data). TrackingTransformer feeds z_{t-1} as input at each step, giving the model a continuously updated view of the beam state. DualStreamTransformer keeps element and beam tokens in separate streams connected by cross-attention, which is a structural middle ground. Either architecture could in principle fix the z₀-only conditioning limitation.

**Methodology:** Comparing architectures under a fixed config (e.g., same bs and lr as prior Lattice runs) is problematic for two reasons. First, bs and lr interact: changing bs without rescaling lr changes the effective optimization trajectory. Second, lr may depend on architecture (different forward pass structures have different gradient magnitudes) and bs depends on model memory footprint. A fair comparison requires each architecture to be evaluated at its own optimal (bs, lr). The additional tuning cost is two 4-GPU jobs per architecture (one bs benchmark, one lr scan) before the main d_model scan.

**Batch size selection:** bs is primarily a hardware/throughput decision — the largest bs that saturates the GPU without OOM. Measured via debug-QOS speed runs (100 steps each): find the bs where samples/sec plateaus. This is architectural (memory footprint) but not strongly dependent on d_model within a reasonable range, so we tune once at d128 and verify d512 doesn't OOM.

**Learning rate selection:** Given the chosen bs, run a lr scan over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} at 200 epochs with ReduceOnPlateau scheduler. The linear scaling rule (lr ∝ bs) gives a starting point when bs differs from the Lattice baseline (bs=32, lr=3e-4), but the rule is approximate and architecture-dependent, so we scan rather than trust the formula. Note: lr may also depend on d_model (gradient magnitudes change with width). The principled fix is μP, which makes hyperparameters transfer exactly across scales. We are not implementing μP for now; instead we tune at d128 (our expected sweet spot) and apply those values across the d_model scan, accepting that results at extreme widths may be slightly suboptimal.

**Why ROP for the lr scan (not cosine):** Using cosine with T_max=150 for a quick scan and then cosine with T_max=500 for the full run creates a systematic bias. With T_max=150, the lr decays to ~0 by the end of the scan, fully consuming the schedule in 150 steps. With T_max=500, the same lr would still be at ~80% of its peak at epoch 150. This means the short cosine scan penalizes higher learning rates — they get fewer effective high-lr steps relative to the full-run schedule. ReduceOnPlateau has no T_max and responds to actual validation progress, so the results transfer cleanly to cosine runs of any length.

**TrackingTransformer — scheduled sampling cost:** The AR benchmark (job 51673422) confirmed two things. First, a bug in `TrackingTransformer._forward_sequential` caused an autograd crash when running the AR backward pass: in-place buffer writes (`z_prev_buf[:, t] = z_cur`, `z_pred_buf[:, t] = z_predicted`) invalidated the computation graph for earlier steps. Fixed by switching to a list + `torch.stack` approach, matching the correct implementation already used by DualStreamTransformer. Second, the DualStream AR benchmark (job 51672840, run during the Pydantic fix wait) confirmed AR training is ~26× slower than teacher forcing (1.76 it/s vs ~45 it/s at bs=32). At 282 batches/epoch, a fully AR 500-epoch training would take ~22 hours — impractical for a 4-hour job. **Decision: use teacher-forcing only (`ss_warmup=500`) for all architecture comparison runs.** This keeps training fast and keeps the comparison fair. Whether scheduled sampling actually improves AR generalization is left as an open question.

**Batch size selection results:** Benchmarks at bs=64/128/256 (jobs 51673586–88) with TrackingTransformer in teacher-forcing mode:

| bs  | Peak it/s | Samples/s |
|-----|-----------|-----------|
| 64  | ~46       | ~2950     |
| 128 | ~30       | ~3840     |
| 256 | ~15       | ~3840     |

Throughput plateaus at bs=128; going to 256 gives no additional benefit. **Selected bs=128** for both TrackingTransformer and DualStreamTransformer.

**LR scan 1 (flawed, cosine, discarded):** Jobs 51676485/51676487, d_model=128, bs=128, 150 epochs, cosine. Trend monotonically improving 1e-4→3e-3 for both architectures (Tracking: 0.01175→0.00310; DualStream: 0.01172→0.00328). All runs fully converged by epoch 150. However, cosine T_max=150 compresses the schedule relative to the 500-epoch full runs, biasing the comparison against higher lrs. Discarded in favor of ROP scan.

**LR scan 2 (corrected, ROP):** Jobs 51680135/51680137, d_model=128, bs=128, 200 epochs, ReduceOnPlateau (factor=0.5, patience=10). Range {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}. Both fully converged by epoch 200. Results:

**TrackingTransformer (TF val_loss — teacher-forcing):**

| lr   | val_loss | train_loss |
|------|----------|------------|
| 1e-4 | 0.00893  | 0.00901    |
| 3e-4 | 0.00602  | 0.00563    |
| 1e-3 | 0.00365  | 0.00284    |
| 3e-3 | 0.00300  | 0.00175    |
| **1e-2** | **0.00271**  | 0.00148    |
| 3e-2 | 0.00342  | 0.00277    |

**DualStreamTransformer (TF val_loss — teacher-forcing):**

| lr   | val_loss | train_loss |
|------|----------|------------|
| 1e-4 | 0.00760  | 0.00727    |
| 3e-4 | 0.00534  | 0.00425    |
| 1e-3 | 0.00343  | 0.00192    |
| **3e-3** | **0.00320**  | 0.00149    |
| 1e-2 | 0.00430  | 0.00341    |
| 3e-2 | 0.04095  | 0.03941    |

**Selected lrs: Tracking → 1e-2, DualStream → 3e-3.**

The two architectures have different optimal lrs, which is expected given their different forward-pass structures. For Tracking, improvement is monotone through 1e-2 with a sweet spot there (3e-2 degrades slightly, suggesting instability at very high lr). For DualStream, the curve peaks at 3e-3 and degrades sharply at 1e-2 and collapses entirely at 3e-2. The cross-attention mechanism likely produces smaller gradient scales at the beam–element interface, making DualStream more sensitive to high lrs.

**Critical methodological note:** The val_loss values above are teacher-forcing losses — the model predicts z_t given the true z_{t-1} at each step. This is fundamentally a different metric than the LatticeTransformer's val_loss, which is always computed open-loop (predicting the full trajectory from z₀ with no intermediate ground truth). The ~0.0027–0.0032 figures are not directly comparable to the Lattice baseline of 0.023. They measure one-step prediction accuracy rather than cumulative trajectory accuracy. The true comparison requires evaluating all architectures in autoregressive inference mode and comparing per-step MSE curves, which is the purpose of the upcoming d_model scan + evaluation.

**Steps:**
1. [done] Speed benchmarks — AR mode (51673422), bs=64/128/256 (51673586–88), DualStream fix (51672840)
2. [done] Fixed inplace autograd bug in `TrackingTransformer._forward_sequential`; decided TF-only training
3. [done] Selected bs=128 from throughput plateau
4. [done] Corrected lr scan with ROP — Tracking best lr=1e-2, DualStream best lr=3e-3
5. [done] d_model scan at tuned (bs=128, best lr), 500 epochs, cosine on `encoded_sectioned_10k`
6. [done] Stable-lr reruns for Tracking d256/d512 (lr=3e-3) and DualStream d512 (lr=1e-3) — jobs 51736765/66/71
7. [pending] Compare AR-mode per-step MSE curves across all three architectures at each d_model

**d_model scan results (teacher-forcing val_loss; Tracking d256/d512 and DualStream d512 from stable-lr reruns):**

| d_model | Tracking (lr) | val_loss | DualStream (lr) | val_loss |
|---------|--------------|----------|----------------|----------|
| 64      | 1e-2         | 0.002850 | 3e-3           | 0.003298 |
| 128     | 1e-2         | 0.002209 | 3e-3           | 0.002789 |
| 256     | 3e-3         | 0.002162 | 3e-3           | 0.002404 |
| 512     | 3e-3         | **0.002096** | 1e-3       | 0.002493 |

**Results and analysis:**

TrackingTransformer wins at every d_model, with a consistent 15–25% lower TF val_loss than DualStreamTransformer. The Tracking advantage is present from d64 and does not diminish at larger widths, suggesting the architectural difference (conditioning on z_{t-1} at each step vs cross-attention over element tokens) is systematically beneficial and not a tuning artifact.

For TrackingTransformer, the best result is d512 at lr=3e-3 (0.002096, epoch 474 of run `scan_arch_tracking_d_model512_lr3e-3_260418_1324`). The improvement over d256 (0.002162) is modest (~3%), consistent with mild overfitting at 10k samples — the train/val gap at d512 (train=0.000296, val=0.002096, ratio ~7×) is much larger than at d128 (ratio ~2.5×). Capacity is no longer the bottleneck; dataset size is.

For DualStreamTransformer, improvement is monotone through d256 (0.002404). The stable d512 rerun at lr=1e-3 gives 0.002493 — slightly worse than d256. The lr=1e-3 is likely too conservative for d512 (optimal lr sits between 1e-3 and 3e-3), but we did not tune further. DualStream d256 (`scan_arch_dualstream_d_model256_260418_0738`) remains the best DualStream checkpoint.

**Training instability in Tracking d256 and d512 at lr=1e-2 (identified and resolved 2026-04-18):**

The initial Tracking d256 and d512 runs used lr=1e-2, which caused repeated gradient explosions with cosine T_max=500. At d256: multiple moderate spikes, worst at epoch 58. At d512: catastrophic spike at epoch 42 — train jumps from 0.003533 to 0.043605 and stays elevated for ~13 epochs; additional large spikes at epochs 130, 186, 202–206, 213. Gradient clipping (clip=1.0) was insufficient. The root cause is that lr=1e-2 was tuned with ROP, which decays lr reactively and effectively lowers the early-phase lr — with cosine T_max=500 the lr holds near 1e-2 for the first ~50 epochs. Stable reruns at lr=3e-3 (jobs 51736765/66) ran spike-free and improved the d512 convergence floor from 0.002161 to 0.002096; d256 was essentially unchanged (0.002138 → 0.002162), confirming the moderate spikes there had little impact.

**Caveat on cross-architecture comparison:** These TF val_losses reflect one-step prediction accuracy (model predicts z_t given true z_{t-1}) and are not directly comparable to LatticeTransformer val_loss, which is a full open-loop trajectory loss from z₀. TrackingTransformer and DualStreamTransformer are ~8–10× lower on their metric, but this largely reflects the easier task structure of teacher forcing. The true comparison — which architecture produces the most accurate AR trajectories — requires evaluating all three in autoregressive inference mode.

A secondary comparison between Tracking and DualStream TF losses is valid (both use the same metric) and consistently favors Tracking. Whether Tracking's advantage holds in AR inference is the key remaining question.

---

## Scan 6 — AR inference comparison (2026-04-18)

**Question:** Which architecture produces the most accurate open-loop trajectories? The TF val_loss comparison in Scan 5 was not directly comparable across architectures (Tracking/DualStream use one-step TF loss; Lattice uses full open-loop loss). Evaluating all three in AR inference mode puts them on equal footing.

**Checkpoints evaluated** (job 51750686, `eval_compare_ar/`):
- TrackingTransformer d512, lr=3e-3, epoch 474 (`scan_arch_tracking_d_model512_lr3e-3_260418_1324`)
- DualStreamTransformer d256, epoch 480 (`scan_arch_dualstream_d_model256_260418_0738`)
- LatticeTransformer d128 L12, epoch 465 (`scan_direct_depth_d128_L12_260414_1314`)

All evaluated on `encoded_sectioned_10k` val split (1k samples, seed=42).

**Results:**

| Model | AR mean MSE | AR final-step MSE | TF mean MSE | AR/TF ratio |
|-------|------------|-------------------|-------------|-------------|
| Tracking d512  | **0.01596** | **0.04410** | 0.002096 | 7.6× |
| DualStream d256 | 0.01778 | 0.04846 | 0.002404 | 7.4× |
| Lattice d128 L12 | 0.02234 | 0.05904 | — (open-loop by design) | — |

**Analysis:**

TrackingTransformer holds its lead in AR mode. It achieves 10% lower AR mean MSE than DualStream (0.01596 vs 0.01778) and 29% lower than Lattice (0.02234). The rank ordering from TF training — Tracking > DualStream > Lattice — transfers cleanly to AR inference, confirming the result is not an artifact of the teacher-forcing metric.

The most striking finding is the near-identical AR/TF degradation ratio for both autoregressive models: Tracking degrades 7.6× and DualStream 7.4×. These ratios are indistinguishable given the model size difference (d512 vs d256) and architecture difference. This means neither architecture has a structural advantage in controlling error accumulation — they simply inherit their AR performance from their TF accuracy floor. The better TF model (Tracking) becomes the better AR model by the same proportional margin.

Lattice's AR mean MSE (0.02234) exactly matches its training val_loss (0.022340), as expected — Lattice is trained open-loop and the val_loss is already the AR metric. The autoregressive models in AR mode (Tracking: 0.01596, DualStream: 0.01778) both beat Lattice despite Lattice using a much larger model (d128 L12 vs d256/d512 for the others). The architectural advantage of conditioning on the previous beam state is real and survives the distribution-matched evaluation.

**Physical explanation for growing MSE at larger element indices.** The per-step MSE curves grow monotonically with element index in both AR *and* TF modes. TF mode uses ground-truth history, so distribution shift cannot explain TF degradation — the physics itself becomes harder at larger indices. The `encoded_sectioned_10k` dataset uses multi-section lattices (2–3 FODO sections per sample), and at section boundaries the beam envelope is mismatched to the new section's optics. This mismatch drives beta beating and irregular beam evolution that is genuinely more difficult to predict regardless of training method. The growing MSE at later elements is therefore the sum of two distinct components: (1) intrinsically harder physics near section boundaries, and (2) distribution shift from teacher forcing. Fixing the training method will reduce component 2 but not flatten the curve entirely. Evaluating on `encoded_sectioned_1sec_10k` (no section boundaries) would isolate component 2 and give a cleaner read on how much of the degradation is attributable to training method alone.

**The AR degradation is driven by outliers, not uniform decay.** The updated plot (mean + median + 10–90 band) shows that the AR median tracks the TF mean closely at every element index. The typical sample in AR mode performs nearly as well as the typical sample under teacher forcing; the 7.5× mean degradation is driven by a small tail of catastrophically failing samples. The mean is not a useful summary of AR performance — the median is.

**Next step: identify and characterize the outliers.** Before attributing the failures to section mismatch or training distribution shift, we need to understand which samples blow up and why. The dataset contains lattices with qualitatively different physics — purely linear (dipoles + quadrupoles) vs nonlinear (sextupoles, RF cavities) — and section counts vary from 1 to 3. Nonlinear dynamics and section-boundary mismatch are both plausible outlier causes and may require different fixes. A dedicated diagnostic session is needed to examine which samples fail and what element features characterize them.

The next step is a dedicated diagnostic session to identify which samples fail and what element features (nonlinear elements, section count) characterize them. The intervention — noise injection, AR fine-tuning, scheduled sampling, or more nonlinear-element-rich data — depends on that result.

---

## Outlier Diagnosis — RF Cavities as Source of Extreme AR MSE (2026-04-22)

**Question:** Which samples drive the large mean/median gap in AR MSE, and what causes them to fail?

Scan 6 established that the AR mean MSE is dominated by a small tail of catastrophically failing samples — the median AR MSE tracks the TF mean closely, making the mean a misleading summary. The next step is to characterize those outliers.

**Finding 1: AR outliers all fail at RF cavities, not sextupoles or other nonlinear elements.** Inspecting the top-10 worst AR samples, every one shows an error jump of at least two orders of magnitude at an RF cavity. The element-type strip in the outlier trajectory plots shows no other common structure — element type at the spike position is RF in all cases.

**Finding 2: TF error also jumps at RF.** This is the decisive observation. In teacher-forcing mode the model receives the true previous beam state at every step, so AR error accumulation cannot explain the failure. The TF error jumping at the same RF slots means the model fails to predict a single RF step correctly even when given the correct input. The mean/median gap in AR MSE is therefore not primarily an AR stability problem — it is a single-step prediction failure at RF elements that AR inference then amplifies.

**Finding 3: RF cavities are frequent (50% of sequences) but sparse within sequences.** Of the 10,000 training sequences, 4,992 contain at least one RF cavity. Among RF-containing sequences, the mean is 2 RF elements out of 32 — RF occupies only 3.1% of element slots (9,950 out of 320,000). Because the trajectory MSE loss averages over all steps equally, RF steps contribute ~3% of the training signal. The model minimizes loss on the 97% non-RF majority and underfits RF specifically. This explains why a model that performs well in aggregate — low TF val_loss, low AR median — still fails catastrophically at RF.

**Finding 4: Most RF steps are predicted fine; only ~1% are catastrophic.** The TF MSE distribution at RF slots has median ≈ 0.0003 (well within normal range) but p99 ≈ 0.48 — a bimodal structure. The model has learned the RF map for most configurations and fails completely for a small subset. The failures are not uniformly distributed across the RF parameter space: scatter plots of TF MSE vs V_rf and phi_rf show no clustering. Failures appear at all voltage levels and all phases.

**Finding 5: `sigma_z × f_rf × V_rf` is the best linear predictor of TF MSE at RF (r ≈ 0.54), but is not sufficient.** We systematically searched for the beam-state and element-parameter features that predict RF failure. The longitudinal bunch length sigma_z of the incoming beam is the strongest single predictor (r = 0.44), and it improves when combined with f_rf (r = 0.51) and further with V_rf (r = 0.54). The physical interpretation is natural: `sigma_z × f_rf` is the bunch length in units of the RF wavelength — how much of the sinusoidal voltage curve the bunch spans — and `V_rf` is the kick strength. When the bunch is long relative to the wavelength and the kick is strong, the energy gain varies strongly across the bunch, and a single latent step cannot represent this differential kick. The RF phase phi_rf and effective phase at the bunch centroid add nothing (r ≈ 0.02 independently), meaning phase is uniformly distributed across failures and is not the discriminator. Interaction terms involving phi_rf also degraded the correlation slightly.

Despite reaching r = 0.54 with the best composite feature, ~70% of the variance in log TF MSE at RF remains unexplained by scalar marginals of the beam state and element parameters. The remaining failures likely depend on the full 6D distribution shape — particularly z–δ coupling or transverse–longitudinal correlations in the latent vector — rather than any individual observable. We did not find a parameter regime that cleanly separates successes from failures.

**Conclusion.** The AR outlier problem is fundamentally a single-step RF prediction problem. The model fails for ~1% of RF slots, and those failures propagate through AR inference to produce the catastrophic tail that inflates the mean MSE ~10× above the median. The condition `sigma_z × f_rf × V_rf` large is necessary but not sufficient for failure, and no simple threshold cleanly identifies which RF steps will fail. The most direct interventions are (1) upweighting RF steps in the training loss so they receive more gradient signal relative to their 3% frequency share, and (2) generating data with more RF-dense lattices so the model encounters more diverse (beam state, RF parameters) combinations during training.

Scripts: `scripts/analyze_ar_outliers.py`, `scripts/analyze_rf_regime.py`, `scripts/analyze_rf_beam_state.py`. Outputs in `runs/scan_arch_tracking_d_model512_lr3e-3_260418_1324/eval_ar_outliers/` and `runs/scan_arch_dualstream_d_model256_260418_0738/eval_ar_outliers/`.

---

## Best Checkpoints

| Architecture | Dataset | Run | TF val_loss | AR mean MSE |
|---|---|---|---|---|
| TrackingTransformer d512 L6 | encoded_sectioned_10k | `scan_arch_tracking_d_model512_lr3e-3_260418_1324` (epoch 474) | 0.002096 | 0.01596 |
| DualStreamTransformer d256 L6 | encoded_sectioned_10k | `scan_arch_dualstream_d_model256_260418_0738` (epoch 480) | 0.002404 | 0.01778 |
| LatticeTransformer d128 L12 | encoded_sectioned_10k | `scan_direct_depth_d128_L12_260414_1314` (epoch 465) | 0.022340 | 0.02234 |
| LatticeTransformer d128 L6 | encoded_sectioned_1sec_10k | `lbd_d128_L6_260414_201843` (epoch 389) | 0.021947 | — |

---

## Pending Experiments

None currently.

---

## Open Questions

- **How to improve AR robustness at RF elements.** Single-step TF failure at RF is the root cause of the outlier tail (diagnosed 2026-04-22). The two most direct interventions are RF loss upweighting and generating data with higher RF density / wider parameter ranges. Whether these are sufficient, or whether a specialized RF module is needed, is open.
- **Is the transformer over history necessary?** Beam dynamics are Markovian; causal attention may be wasted capacity. Revisit after outlier analysis and robustness work are resolved.
- **How to scale the dataset beyond 10k samples.** Raw tracking output is ~200 MB/sample (33 snapshots × 100k particles × 6D × float64), making 100k samples ~20 TB — exceeding scratch quota. Requires a redesigned pipeline that encodes and deletes raw particle data incrementally rather than accumulating the full dataset before encoding.

