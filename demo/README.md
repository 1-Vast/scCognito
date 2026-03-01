# Downstream Demos

Embedding-based downstream tasks are implemented in `demo/run_downstream.py`.

## CLI

```bash
python -m demo.run_downstream \
  --embedded_h5ad outputs/run_x/plm_outputs/plm_embedded.h5ad \
  --out_dir outputs/run_x/downstream_outputs \
  --emb_key X_plm \
  --n_domains 8 \
  --label_col cell_type \
  --batch_col batch \
  --time_col timepoint \
  --perturb_genes CD3D,MS4A1
```

## Outputs

- `downstream_metrics.json`
- `downstream_report.html`
- `downstream_domain_assignments.csv`
- `plm_embedded_with_downstream.h5ad`

## Implemented tasks

- Spatial domain identification (clustering + silhouette/DB + optional ARI/NMI + spatial coherence)
- Cross-representation alignment (distance correlation with other `obsm` views)
- Large-scale integration diagnostics (batch entropy + batch silhouette)
- Temporal evolution check (pseudo-time vs `time_col` Spearman rho)
- Computational perturbation proxy (gene-set score + sensitive domains)
- Mechanistic interpretability (domain marker genes)
- Cognitive feedback and autonomous next-action notes
