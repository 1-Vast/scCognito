# Pipeline & Artifacts (Ground Truth)
Keywords:
pipeline, artifacts, outputs, embedded h5ad,
metrics fields, verification, workspace

---

## Overall Loop

Teacher
→ SER / Bridge
→ PLM Training
→ Embedded h5ad Export
→ Agent Analysis
→ Next Configuration

---

## Key Artifacts (MUST NOT INVENT)

Teacher output:
- JSON tokens file

PLM output:
- plm_embedded.h5ad

---

## Embedded h5ad Required Fields (obs)

- plm_recon_err
- plm_ser_energy
- ser_coverage
- ser_strength

Missing fields = invalid output.

---

## Workspace Rules

OUT_ROOT defined by `.env`

Agent treats OUT_ROOT as workspace root.

---

## Safety

If file missing:
DO NOT GUESS.

Instead:
- verify using filesystem tools
- request validation