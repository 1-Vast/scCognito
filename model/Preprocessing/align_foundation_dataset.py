import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from tqdm import tqdm

DATA_DIR = r"D:\LLM\data\Foundation"
FIX_DIR = os.path.join(DATA_DIR, "fixed")  
ALIGN_DIR = os.path.join(DATA_DIR, "Foundation_train_aligned")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.tsv")

os.makedirs(FIX_DIR, exist_ok=True)
os.makedirs(ALIGN_DIR, exist_ok=True)

EXCLUDE = {
    "Gene_classification.h5ad",
}

h5ads = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
h5ads = [p for p in h5ads if os.path.basename(p) not in EXCLUDE]

print(f"\nFound {len(h5ads)} datasets")

def to_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

def fix_var_index_conflict(adata: ad.AnnData):
    adata.var.index = adata.var.index.astype(str)
    adata.var_names = adata.var_names.astype(str)

    idx_name = adata.var.index.name
    if idx_name is not None and idx_name in adata.var.columns:
        col = adata.var[idx_name].astype(str).values
        idxv = adata.var.index.astype(str).values

        if np.array_equal(col, idxv):
            adata.var.drop(columns=[idx_name], inplace=True)
        else:
            adata.var.rename(columns={idx_name: f"{idx_name}_col"}, inplace=True)

    adata.var.index.name = None

print("\n[1/3] Fixing original h5ad files (obs_names/var_names + var conflicts)...")

fixed_paths = []
skipped_fix = []

for p in tqdm(h5ads, desc="Fixing"):
    name = os.path.splitext(os.path.basename(p))[0]
    out = os.path.join(FIX_DIR, f"{name}.fixed.h5ad")

    try:
        adata = sc.read_h5ad(p)

        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        adata.var_names = adata.var_names.astype(str)
        adata.var.index = adata.var.index.astype(str)

        fix_var_index_conflict(adata)

        adata.write_h5ad(out, compression="gzip")
        fixed_paths.append(out)

    except Exception as e:
        skipped_fix.append((p, str(e)))
        print(f"\n⚠ FIX SKIP: {p}\n    reason: {e}\n")

print(f"\nFixed saved: {len(fixed_paths)} files -> {FIX_DIR}")
if skipped_fix:
    print(f"Skipped during fixing: {len(skipped_fix)} files (see messages above)")

print("\n[2/3] Building global gene vocabulary from fixed files...")

genes = set()
for p in tqdm(fixed_paths, desc="Scanning genes"):
    try:
        a = sc.read_h5ad(p, backed="r")
        a.var_names = a.var_names.astype(str)
        genes.update(a.var_names.tolist())
        a.file.close()
    except Exception as e:
        print(f"\n⚠ VOCAB SKIP: {p}\n    reason: {e}\n")

vocab = sorted(genes)
pd.Series(vocab).to_csv(VOCAB_PATH, sep="\t", index=False, header=False)
print(f"Vocabulary saved: {VOCAB_PATH}")
print("Total genes:", len(vocab))

vocab_index = {g: i for i, g in enumerate(vocab)}
G = len(vocab)

print("\n[3/3] Aligning fixed datasets -> aligned datasets...")

skipped_align = []

for p in tqdm(fixed_paths, desc="Datasets"):
    name = os.path.splitext(os.path.basename(p))[0].replace(".fixed", "")
    out_path = os.path.join(ALIGN_DIR, f"{name}.aligned.h5ad")

    try:
        adata = sc.read_h5ad(p)

        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        adata.var_names = adata.var_names.astype(str)
        adata.var.index = adata.var.index.astype(str)
        fix_var_index_conflict(adata)

        X = to_csr(adata.X)
        n = adata.n_obs

        cols_src = []
        cols_dst = []

        # mapping
        for j, g in enumerate(adata.var_names):
            idx = vocab_index.get(g)
            if idx is not None:
                cols_src.append(j)
                cols_dst.append(idx)

        if not cols_src:
            print(f"\n⚠ ALIGN SKIP: {name} has 0 matched genes to vocab.\n")
            continue

        X_sub = X[:, cols_src]
        k = len(cols_src)

        P = sp.csr_matrix(
            (np.ones(k, dtype=np.float32), (np.arange(k), np.array(cols_dst))),
            shape=(k, G)
        )
        X_new = X_sub @ P  # (n x G)

        var = pd.DataFrame(index=pd.Index(vocab, dtype=str))
        aligned = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var)
        aligned.obs["source_dataset"] = name

        aligned.write_h5ad(out_path, compression="gzip")

    except Exception as e:
        skipped_align.append((p, str(e)))
        print(f"\n⚠ ALIGN SKIP: {p}\n    reason: {e}\n")

print("\n✅ Done.")
print("Fixed dir :", FIX_DIR)
print("Aligned dir:", ALIGN_DIR)
print("Vocab path:", VOCAB_PATH)

if skipped_align:
    print(f"\nSkipped during aligning: {len(skipped_align)} files (see messages above)")