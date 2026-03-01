# Framework Terminology Mapping (Operational)
Keywords:
framework, SER, lam_ser, bridge, embedding,
PLM, LLM, alignment, semantic energy, math format

Purpose:
Provide executable meaning of concepts and parameters.

---

## System Structure

PLM -> Structural World Model
LLM -> Semantic Reasoning Engine
Agent -> Decision Controller

Loop:
Teacher -> Bridge -> PLM -> Metrics -> Agent

---

## Semantic Energy Regularization (SER)

Definition:
Semantic constraint applied to embedding manifold.

Canonical math format (for HTML reports):
Use standard notation:
$L_{\text{total}} = L_{\text{structure}} + \lambda \cdot E_{\text{semantic}}$

Parameter:
lam_ser = semantic control weight

Interpretation:
Higher lam_ser -> stronger semantic prior
Lower lam_ser -> freer structural learning

---

## Bridge Layer

Transforms:
embedding -> semantic summary

Outputs:
- uncertainty statistics
- structure summaries
- prompts for teacher

---

## Metrics Used By Agent

Agent decisions MUST rely on measurable metrics:

- plm_recon_err
- plm_ser_energy
- ser_coverage
- ser_strength

Agent MUST NOT claim success without reading the exported metrics.