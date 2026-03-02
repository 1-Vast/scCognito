import scanpy as sc
import numpy as np

adata = sc.read_h5ad("D:/LLM/outputs/DLPFC_V2/plm_outputs/plm_embedded.h5ad")

Z = adata.obsm["X_plm"] 
obs_names = adata.obs_names.to_numpy()
spatial = adata.obsm["spatial"] if "spatial" in adata.obsm else None

out_npz = "D:/LLM/outputs/DLPFC_V2/plm_outputs/plm_features.for_cluster.npz"
np.savez(out_npz, embedding=Z, obs_names=obs_names, spatial=spatial)
print(f"Exported features to {out_npz}")