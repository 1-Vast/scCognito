# Tuning Knowledge Base
Keywords:
tuning, diagnose, instability, recon_err, ser_energy,
coverage, strength, parameter adjustment,
training failure, optimization, step size

Each block is SELF-CONTAINED.
Any chunk alone must enable correct action.

---

## Global Rule: Small-Step Tuning

Keywords:
step size, incremental, conservative, avoid oscillation

Rule:
Prefer small incremental changes unless evidence clearly supports a large jump.

Recommended step sizes:
- mask_ratio: <= 0.05 per iteration
- lam_ser: <= 0.05 per iteration

---

## Issue: Low SER Coverage

Keywords:
ser_coverage, weak supervision, semantic missing,
low coverage, insufficient tokens

Condition:
- ser_coverage p10 <= 1
OR
- ser_strength p10 < 0.3

Rationale:
Semantic information fails to propagate into the embedding manifold.

Action:
- Reduce bridge parameter `conf_floor` slightly
  Example: 0.60 -> 0.55 -> 0.50
- If persists: rerun teacher

Risk:
Too low conf_floor introduces noisy supervision.

---

## Issue: High Reconstruction Error (plm_recon_err)

Keywords:
plm_recon_err, reconstruction error, masking instability,
error tail, training unstable, mask_ratio

Condition:
- recon_err p99 >> mean
- typically > 5× mean

Rationale:
Model struggles to reconstruct structural world model.

Likely causes:
- excessive masking
- semantic constraint too strong

Action:
- Reduce `mask_ratio` in small steps
  Example: 0.30 -> 0.25
- Optionally reduce `lam_ser` in small steps
  Example: 0.50 -> 0.45 -> 0.40

Expected Effect:
Improved structural stability.

---

## Issue: High Semantic Energy Tail

Keywords:
ser_energy, semantic overload,
over-regularization, energy explosion

Condition:
- ser_energy p99 >> mean
- typically > 4× mean

Rationale:
Semantic loss dominates structural learning.

Action:
- Reduce `lam_ser` gradually
  Example: 0.40 -> 0.35 -> 0.30
- Retrain PLM

---

## Issue: Weak Learning / Metric Plateau

Keywords:
flat metrics, stagnation, no improvement,
training plateau

Condition:
- narrow metric distribution
- clustering unchanged

Action:
- Slightly increase lam_ser (small step)
- Rerun teacher for stronger tokens

---

## Issue: Over-Smoothing Domains

Keywords:
over smooth, spatial blur, lost boundaries

Condition:
- domains overly continuous
- biological boundaries disappear

Action:
- Adjust mask_ratio in small steps
- Reduce smoothing-related parameters