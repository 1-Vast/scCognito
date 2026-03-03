# scCognito: LLM-Agent Guided Spatial Transcriptomics Closed-Loop System

scCognito is a closed-loop spatial transcriptomics framework:

**Teacher → Bridge → PLM → Agent**

It integrates:
- an LLM **Teacher** for semantic prior extraction,
- a **Bridge** that maps priors into semantic energy representations (SER),
- a **PLM** (graph + attention hybrid) for robust embeddings,
- a strict **Agent** that monitors metrics and proposes next-round configs.

## Key Features

- **Strict tool-calling Agent** with protocol-safe context pruning and deterministic fallback
- **Dual-stream graph encoder** (spatial graph + attribute graph) with **node-adaptive fusion**
- **Talking-Heads global attention** to capture long-range tissue-level dependencies
- **Self-supervised contrastive option** (cross-view InfoNCE) for stronger global structure

## 1.Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

---

## 2.Required LLM Environment

Create `model/teacher/.env.teacher`:

```env
TEACHER_API_KEY=...
TEACHER_BASE_URL=https://.../api/v3
TEACHER_MODEL_ID=...
```

Optional: root `.env` for agent:

```env
AGENT_API_KEY=...
AGENT_BASE_URL=https://.../api/v3
AGENT_MODEL_ID=...
AGENT_MAX_TURNS=6
AGENT_MAX_HISTORY_MESSAGES=12
```

If AGENT_* is missing, Agent falls back to TEACHER_*.

---

## 3. Terminal usage guide

### 3.1 Run the full pipeline

```bash
python -m main \
  --h5ad data/demo/DLPFC/151673.h5ad \
  --out_root outputs/dlpfc_151673 \
  --groupby sce.layer_guess \
  --device cuda \
  --max_llm_calls 24 \
  --epochs 1200 \
  --d_hid 512 \
  --d_out 256 \
  --n_layers 3
```

Key terminal markers:
- `[PROGRESS][PIPELINE] stage=... pct=...`
- `[PROGRESS][PLM] epoch=.../.... pct=... loss=...`

### 3.2 Run downstream tasks on embedding

```bash
python -m demo.run_downstream \
  --embedded_h5ad outputs/dlpfc_151673/plm_outputs/plm_embedded.h5ad \
  --out_dir outputs/dlpfc_151673/downstream_outputs \
  --emb_key X_plm \
  --n_domains 7 \
  --label_col sce.layer_guess
```

### 3.3 Run agent report (optional)

```bash
set PYTHONPATH=model
python -m agent.cli \
  --goal "Generate optimization report" \
  --h5ad data/demo/DLPFC/151673.h5ad \
  --groupby sce.layer_guess \
  --out_root outputs/dlpfc_151673
```

---

## 4. Web usage guide (recommended)

### 4.1 Start web server

```bash
python -m uvicorn web_portal.app:app --host 127.0.0.1 --port 8000 --log-level debug
```

Open:

`http://127.0.0.1:8000`

### 4.2 Fill fields (DLPFC one-slice example)

- `h5ad`: `data/demo/DLPFC/151673.h5ad`
- `out_root`: `outputs/web_dlpfc_151673`
- `groupby`: `sce.layer_guess`
- Teacher LLM fields (or rely on `.env.teacher`)

### 4.3 Set training config (PLM Train Config)

Recommended initial values:
- `epochs=200`
- `lr=0.001`
- `weight_decay=0.0001`
- `d_hid=256`, `d_out=128`, `n_layers=2`, `dropout=0.1`
- `grad_clip=5.0`, `log_every=10`
- `w_recon=1.0`, `w_spatial_pred=1.0`, `ser_w_proto=1.0`

### 4.4 Set downstream config

- `n_domains=7` (DLPFC layers often around this scale)
- `label_col=sce.layer_guess` (if present)
- `batch_col/time_col` optional
- `emb_key=X_plm`
- `perturb_genes` optional (comma-separated)

### 4.5 Click order

1. `Run Main`
2. `Run Downstream`
3. `Run Agent` (optional)
4. `Run Auto-Loop` (optional)

### 4.6 Progress and logs

Web shows:
- Pipeline progress bar
- PLM epoch progress bar
- SSE live logs

---

## 5. Outputs

Main outputs under `out_root`:
- `teacher_outputs/`
- `ser_outputs/`
- `plm_outputs/`
- `downstream_outputs/`
- `artifacts/`

key files:
- `plm_outputs/plm_ckpt.pt`
- `plm_outputs/plm_ckpt.pt`
- `downstream_outputs/downstream_metrics.json`
- `downstream_outputs/downstream_report.html`
- `next_config.json (auto-loop / agent output)`

---

## 6. Downstream tasks implemented

Based on generated embedding:
- spatial domain identification
- cross-representation alignment
- cognitive feedback for experiment design
- mechanistic interpretability
- computational perturbation proxy
- temporal evolution analysis
- large-scale integration diagnostics
- autonomous next-action suggestions

---

## 7. Troubleshooting

1. `ModuleNotFoundError: fastapi`
- Run `pip install -r requirements.txt`

2. Web cannot call LLM
- Check `.env.teacher` and network access

3. `downstream` fails with missing embedding
- Run `Run Main` first, then `Run Downstream`

4. Ark tool-calling 400
- Agent has deterministic fallback; pipeline still proceeds
