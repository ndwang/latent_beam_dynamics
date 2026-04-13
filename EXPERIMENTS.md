# Experiment Log

All runs use: `data/encoded_sectioned_10k`, `fusion=concat`, `lr=3e-4`, `batch_size=128`, `latent_dim=256`, `n_heads=8`, `mlp_ratio=4`.

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

Runs for L=1–4 all crashed after ~30 seconds (~10–34 epochs). **No usable results.**

**Conclusions:**
- Strong gains from L=1 to L=3; diminishing returns from L=3 onward (L3≈L4, L6 still slightly better).
- L=6 remains the best tested depth.
- The d_model=512 × n_layers sweep needs to be rerun — unclear whether the same saturation pattern holds at larger width.

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

## Best Config (as of 2026-04-06)

```
d_model=512, n_layers=6, n_heads=8, mlp_ratio=4
dropout=0.1, weight_decay=0.01, lr=3e-4
scheduler=cosine, epochs=500, batch_size=128
```

val_loss = **0.0263** (`lbd_d512_L6_260406_185838`)

---

## Open Questions

