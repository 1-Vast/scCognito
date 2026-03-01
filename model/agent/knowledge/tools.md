# Tool Interface Specification
Keywords:
tools, schema, parameters, allowed values,
pipeline.run_main, analyze_embedded,
validation, step size, safe tuning

Agent MUST follow these constraints.

---

## Global Tuning Discipline (Step Size)

Keywords:
step size, small steps, conservative tuning, avoid oscillation

Rule:
Parameter adjustments must be small and incremental unless evidence proves a large jump is required.

Recommended step sizes:
- mask_ratio: change by <= 0.05 per iteration
- lam_ser: change by <= 0.05 per iteration

Invalid behavior:
- large jumps (e.g., mask_ratio 0.30 -> 0.05) without strong evidence
- repeated micro-tweaks (e.g., lam_ser +/- 0.01) without measurable improvement

---

## Tool: pipeline.run_main

Purpose:
Execute PLM training.

Required Parameters:

device:
- allowed: ["cuda", "cpu"]

mask_ratio:
- float
- range: 0.05 – 0.50

lam_ser:
- float
- range: 0.0 – 1.0

Optional:
- run_id (string)

INVALID:
- invent parameters
- unsupported device names

---

## Tool: agent.analyze_embedded

Purpose:
Analyze PLM embedding output.

Parameters:

input_file:
- must exist

groupby:
- default = "leiden"

Allowed:
["leiden", "louvain"]

---

## File Expectations

PLM export MUST contain:

plm_embedded.h5ad

obs fields:
- plm_recon_err
- plm_ser_energy
- ser_coverage
- ser_strength

If missing:
call validation tool.