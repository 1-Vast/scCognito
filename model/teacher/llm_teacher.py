"""
LLM Teacher (Layer A/B/C) - Minimal runnable version.

- Layer A: Token Generation (LLM + structured evidence)
- Layer B: Semantic Filtering (ontology validity + mutual exclusion + evidence gating + granularity downgrade)
- Layer C: Representation Shaping (export constraints, no PLM training here)

All comments are in English as requested.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

COMMON_GROUPBY_KEYS = [
    "cell_type",
    "celltype",
    "CellType",
    "labels",
    "str_labels",
    "annotation",
    "annot",
    "cluster",
    "clusters",
    "seurat_clusters",
    "louvain",
    "leiden",
]

# ---------------------------
# Utilities: file loaders
# ---------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_gmt(path: str) -> Dict[str, Set[str]]:
    gene_sets: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = {g.strip() for g in parts[2:] if g.strip()}
            if genes:
                gene_sets[name] = genes
    return gene_sets


def load_go_obo_ids_and_names(obo_path: str) -> Tuple[Set[str], Dict[str, Dict[str, str]]]:
    go_ids: Set[str] = set()
    meta: Dict[str, Dict[str, str]] = {}

    cur_id = None
    cur_name = None
    cur_ns = None

    for line in _read_text(obo_path).splitlines():
        line = line.strip()
        if line == "[Term]":
            if cur_id and cur_id.startswith("GO:"):
                go_ids.add(cur_id)
                meta[cur_id] = {
                    "name": cur_name or "",
                    "namespace": cur_ns or ""
                }
            cur_id, cur_name, cur_ns = None, None, None
            continue

        if line.startswith("id: "):
            cur_id = line.replace("id:", "").strip()
        elif line.startswith("name: "):
            cur_name = line.replace("name:", "").strip()
        elif line.startswith("namespace: "):
            cur_ns = line.replace("namespace:", "").strip()

    if cur_id and cur_id.startswith("GO:"):
        go_ids.add(cur_id)
        meta[cur_id] = {"name": cur_name or "", "namespace": cur_ns or ""}

    return go_ids, meta


def load_goa_gaf_gene2go(gaf_path: str) -> Dict[str, Set[str]]:
    gene2go: Dict[str, Set[str]] = {}
    with open(gaf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            gene = parts[2].strip()
            go_id = parts[4].strip()
            if not gene or not go_id:
                continue
            gene2go.setdefault(gene, set()).add(go_id)
    return gene2go


# ---------------------------
# Enrichment (evidence builder)
# ---------------------------

@dataclass
class EnrichHit:
    token_type: str            
    token_id: str              
    name: str                  
    p_value: float
    overlap: int
    overlap_genes: List[str]


def _hypergeom_pval(M: int, n: int, N: int, k: int) -> float:
    if k <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, M, n, N))


def enrich_go(
    marker_genes: List[str],
    gene_universe: Set[str],
    gene2go: Dict[str, Set[str]],
    go_allow: Set[str],
    go_meta: Dict[str, Dict[str, str]],
    top_k: int = 8,
) -> List[EnrichHit]:
    markers = [g for g in marker_genes if g in gene_universe]
    marker_set = set(markers)
    if len(marker_set) < 5:
        return []

    go2genes: Dict[str, Set[str]] = {}
    for g in gene_universe:
        for go_id in gene2go.get(g, []):
            if go_id in go_allow:
                go2genes.setdefault(go_id, set()).add(g)

    M = len(gene_universe)
    N = len(marker_set)

    hits: List[EnrichHit] = []
    for go_id, genes_in_term in go2genes.items():
        n = len(genes_in_term)
        overlap_genes = sorted(list(marker_set.intersection(genes_in_term)))
        k = len(overlap_genes)
        if k < 3:
            continue
        p = _hypergeom_pval(M, n, N, k)
        meta = go_meta.get(go_id, {})
        name = meta.get("name", go_id)
        hits.append(
            EnrichHit(
                token_type="GO",
                token_id=go_id,
                name=name,
                p_value=p,
                overlap=k,
                overlap_genes=overlap_genes[:20],
            )
        )

    hits.sort(key=lambda x: (x.p_value, -x.overlap))
    return hits[:top_k]


def enrich_gmt(
    marker_genes: List[str],
    gene_universe: Set[str],
    gmt: Dict[str, Set[str]],
    prefix: str,
    top_k: int = 8,
) -> List[EnrichHit]:
    markers = [g for g in marker_genes if g in gene_universe]
    marker_set = set(markers)
    if len(marker_set) < 5:
        return []

    M = len(gene_universe)
    N = len(marker_set)

    hits: List[EnrichHit] = []
    for set_name, genes_in_set in gmt.items():
        genes_in_set_u = genes_in_set.intersection(gene_universe)
        n = len(genes_in_set_u)
        if n < 10:
            continue
        overlap_genes = sorted(list(marker_set.intersection(genes_in_set_u)))
        k = len(overlap_genes)
        if k < 3:
            continue
        p = _hypergeom_pval(M, n, N, k)
        token_id = f"{prefix}:{set_name}"
        hits.append(
            EnrichHit(
                token_type="Pathway",
                token_id=token_id,
                name=set_name,
                p_value=p,
                overlap=k,
                overlap_genes=overlap_genes[:20],
            )
        )

    hits.sort(key=lambda x: (x.p_value, -x.overlap))
    return hits[:top_k]


# ---------------------------
# Layer A/B/C: Teacher core
# ---------------------------

@dataclass
class Token:
    token_type: str         
    token_id: str           
    name: str
    evidence: Dict[str, Any]
    confidence: float


class LLMSemanticTeacher:
    def __init__(
        self,
        llm_client,
        go_obo_path: str,
        goa_gaf_path: str,
        hallmark_gmt_path: str,
        reactome_gmt_path: str,
        max_llm_calls: int = 64,
        privacy_guard: bool = True,
        debug_checks: bool = False,
    ):
        self.llm = llm_client
        self.max_llm_calls = int(max(1, max_llm_calls))
        self.privacy_guard = bool(privacy_guard)
        self.debug_checks = bool(debug_checks)
        self.llm_calls = 0
        self._limit_warned = False

        self.go_allow, self.go_meta = load_go_obo_ids_and_names(go_obo_path)
        self.gene2go = load_goa_gaf_gene2go(goa_gaf_path)
        self.hallmark = load_gmt(hallmark_gmt_path)
        self.reactome = load_gmt(reactome_gmt_path)

        self.pathway_allow = set()
        self.pathway_allow.update({f"MSIGDB_H:{k}" for k in self.hallmark.keys()})
        self.pathway_allow.update({f"REACTOME:{k}" for k in self.reactome.keys()})

    def _sanitize_evidence_obj(self, evidence_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Privacy guard:
        - keep only aggregate biological evidence
        - never send cell-level matrices or raw per-cell values
        """
        sanitized = {
            "cluster": str(evidence_obj.get("cluster", "")),
            "top_markers": [str(x) for x in evidence_obj.get("top_markers", [])[:25]],
            "enrichment": evidence_obj.get("enrichment", {}),
        }

        if self.privacy_guard:
            lowered = json.dumps(sanitized, ensure_ascii=False).lower()
            banned = ("\"x\"", "matrix", "counts", "expression_matrix", "cell_values", "adata")
            if any(b in lowered for b in banned):
                raise ValueError("Privacy guard detected disallowed raw data fields in teacher evidence payload.")

        return sanitized

    def layer_a_generate_tokens(
        self,
        cluster_name: str,
        top_markers: List[str],
        go_hits: List[EnrichHit],
        hallmark_hits: List[EnrichHit],
        reactome_hits: List[EnrichHit],
        max_tokens: int = 8,
    ) -> List[Token]:
        if self.llm_calls >= self.max_llm_calls:
            if not self._limit_warned:
                print(f"[Teacher][WARN] max_llm_calls reached ({self.max_llm_calls}); remaining clusters will skip LLM calls.")
                self._limit_warned = True
            return []

        evidence_obj = {
            "cluster": cluster_name,
            "top_markers": top_markers[:25],
            "enrichment": {
                "GO": [{"id": h.token_id, "name": h.name, "p_value": h.p_value, "overlap": h.overlap, "genes": h.overlap_genes} for h in go_hits],
                "Hallmark": [{"id": h.token_id, "name": h.name, "p_value": h.p_value, "overlap": h.overlap, "genes": h.overlap_genes} for h in hallmark_hits],
                "Reactome": [{"id": h.token_id, "name": h.name, "p_value": h.p_value, "overlap": h.overlap, "genes": h.overlap_genes} for h in reactome_hits],
            },
        }
        evidence_obj = self._sanitize_evidence_obj(evidence_obj)

        system = (
            "You are a biomedical semantic teacher. "
            "Return ONLY valid JSON, no markdown, no extra text."
        )

        user = {
            "task": "Generate biological semantic tokens for a cell cluster.",
            "requirements": {
                "token_types": ["GO", "Pathway", "CellState", "Phenotype", "Regulation"],
                "must_include_evidence": True,
                "must_include_confidence": True,
                "confidence_range": [0.0, 1.0],
                "max_tokens": max_tokens,
                "strict_schema": {
                    "tokens": [
                        {
                            "token_type": "GO|Pathway|CellState|Phenotype|Regulation",
                            "token_id": "string",
                            "name": "string",
                            "confidence": "float",
                            "evidence": {
                                "markers": "list[string]",
                                "enrichment_refs": "list[object]",
                                "notes": "string"
                            }
                        }
                    ]
                },
                "rule": "If you cannot provide evidence, do NOT output that token."
            },
            "evidence_input": evidence_obj,
        }

        resp_text = self.llm.chat(system_prompt=system, user_content=json.dumps(user, ensure_ascii=False))
        self.llm_calls += 1
        parsed = self._safe_parse_json(resp_text)

        tokens: List[Token] = []
        for t in parsed.get("tokens", []):
            try:
                tokens.append(
                    Token(
                        token_type=str(t.get("token_type", "")),
                        token_id=str(t.get("token_id", "")),
                        name=str(t.get("name", "")),
                        evidence=t.get("evidence", {}) or {},
                        confidence=float(t.get("confidence", 0.0)),
                    )
                )
            except Exception:
                continue
        return tokens

    def layer_b_semantic_filtering(self, tokens: List[Token]) -> List[Token]:
        tokens = self._evidence_gating(tokens)
        tokens = self._ontology_validity(tokens)
        tokens = self._mutual_exclusion(tokens)
        tokens = self._granularity_downgrade(tokens)
        tokens.sort(key=lambda x: x.confidence, reverse=True)
        return tokens

    def layer_c_export_constraints(self, tokens: List[Token]) -> Dict[str, Any]:
        return {
            "prototype_constraints": [
                {"anchor": t.token_id, "type": t.token_type, "confidence": t.confidence}
                for t in tokens if t.confidence >= 0.6
            ],
            "weak_labels": [
                {"label": t.name, "type": t.token_type, "confidence": t.confidence}
                for t in tokens if t.token_type in ("CellState", "Phenotype") and t.confidence >= 0.6
            ],
        }

    def _evidence_gating(self, tokens: List[Token]) -> List[Token]:
        out = []
        for t in tokens:
            ev = t.evidence or {}
            markers = ev.get("markers", [])
            refs = ev.get("enrichment_refs", [])
            if t.confidence < 0.35:
                continue
            if not markers and not refs:
                continue
            out.append(t)
        return out

    def _ontology_validity(self, tokens: List[Token]) -> List[Token]:
        out = []
        for t in tokens:
            if t.token_type == "GO":
                if t.token_id in self.go_allow:
                    out.append(t)
            elif t.token_type == "Pathway":
                if t.token_id in self.pathway_allow:
                    out.append(t)
            else:
                out.append(t)
        return out

    def _mutual_exclusion(self, tokens: List[Token]) -> List[Token]:
        groups = [
            ("B_CELL", ["B cell", "B-cell", "CD79A", "MS4A1"]),
            ("T_CELL", ["T cell", "T-cell", "CD3D", "CD3E", "TRAC"]),
            ("NK", ["NK", "NKG7", "GNLY"]),
            ("MONO", ["Monocyte", "LYZ", "S100A8", "S100A9"]),
        ]

        def score_group(t: Token, kws: List[str]) -> bool:
            s = (t.name + " " + t.token_id).lower()
            return any(k.lower() in s for k in kws)

        kept = []
        used_group = set()

        for t in sorted(tokens, key=lambda x: x.confidence, reverse=True):
            if t.token_type not in ("CellState", "Phenotype"):
                kept.append(t)
                continue

            hit_group = None
            for g, kws in groups:
                if score_group(t, kws):
                    hit_group = g
                    break

            if hit_group is None:
                kept.append(t)
                continue

            if hit_group in used_group:
                continue
            used_group.add(hit_group)
            kept.append(t)

        return kept

    def _granularity_downgrade(self, tokens: List[Token]) -> List[Token]:
        out = []
        for t in tokens:
            refs = (t.evidence or {}).get("enrichment_refs", [])
            if refs and isinstance(refs, list):
                overlaps = []
                for r in refs:
                    try:
                        overlaps.append(int(r.get("overlap", 0)))
                    except Exception:
                        pass
                if overlaps and max(overlaps) < 3 and t.confidence < 0.55:
                    continue
            out.append(t)
        return out

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


# ---------------------------
# runner
# ---------------------------

def build_cluster_markers(adata, groupby: str = "leiden", n_top: int = 25) -> Dict[str, List[str]]:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    selected = False
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
        hv = adata.var["highly_variable"].fillna(False).to_numpy()
        if hv.sum() > 0:
            adata = adata[:, hv].copy()
            selected = True
    except Exception:
        pass

    if not selected:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
            hv = adata.var["highly_variable"].fillna(False).to_numpy()
            if hv.sum() > 0:
                adata = adata[:, hv].copy()
                selected = True
        except Exception as e:
            print(f"[Teacher][WARN] HVG failed; use all genes for marker ranking. reason={e}")

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    if groupby not in adata.obs:
        sc.tl.leiden(adata, resolution=0.8, key_added=groupby)

    if adata.obs[groupby].dtype.name != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    markers: Dict[str, List[str]] = {}

    cats = list(adata.obs[groupby].cat.categories)
    for c in cats:
        df = sc.get.rank_genes_groups_df(adata, group=c)
        genes = df["names"].astype(str).tolist()
        seen = set()
        top = []
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            top.append(g)
            if len(top) >= n_top:
                break
        markers[str(c)] = top
    return markers, adata


def resolve_groupby_key(adata, preferred: str) -> Tuple[str, str]:
    if preferred in adata.obs:
        return preferred, "preferred"
    for k in COMMON_GROUPBY_KEYS:
        if k in adata.obs:
            return k, "common_fallback"
    return "leiden", "computed_leiden"


def run_teacher(
    teacher: LLMSemanticTeacher,
    h5ad_path: str,
    out_dir: str,
    groupby: str = "leiden",
    output_name: str = None,
):
    """
    Generalized entry point to run the Semantic Teacher on any valid .h5ad dataset.
    """
    os.makedirs(out_dir, exist_ok=True)

    adata_raw = sc.read_h5ad(h5ad_path)
    gene_universe = set(adata_raw.var_names.astype(str).tolist())

    resolved_groupby, mode = resolve_groupby_key(adata_raw, groupby)
    if mode == "preferred":
        print(f"[Teacher] using groupby='{resolved_groupby}'")
    elif mode == "common_fallback":
        print(f"[Teacher][WARN] groupby='{groupby}' missing; fallback to common label column '{resolved_groupby}'.")
    else:
        print(f"[Teacher][WARN] no common class label found; computing '{resolved_groupby}' via leiden (in-memory only).")
        adata_temp = adata_raw.copy()
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        try:
            sc.pp.highly_variable_genes(adata_temp, n_top_genes=2000, flavor="seurat_v3")
        except Exception:
            sc.pp.highly_variable_genes(adata_temp, n_top_genes=2000, flavor="seurat")
        adata_temp = adata_temp[:, adata_temp.var["highly_variable"]].copy()
        sc.pp.pca(adata_temp)
        sc.pp.neighbors(adata_temp, n_neighbors=15, n_pcs=30)
        sc.tl.leiden(adata_temp, resolution=0.8, key_added=resolved_groupby)
        adata_raw.obs[resolved_groupby] = adata_temp.obs[resolved_groupby].astype(str).values

    markers_by_cluster, _ = build_cluster_markers(adata_raw.copy(), groupby=resolved_groupby, n_top=25)

    all_results = {}

    for cluster, top_markers in tqdm(
        markers_by_cluster.items(),
        desc=f"Analyzing cluster [{resolved_groupby}]",
        ncols=90,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="green"
    ):
        go_hits = enrich_go(
            marker_genes=top_markers,
            gene_universe=gene_universe,
            gene2go=teacher.gene2go,
            go_allow=teacher.go_allow,
            go_meta=teacher.go_meta,
            top_k=8,
        )
        hallmark_hits = enrich_gmt(
            marker_genes=top_markers,
            gene_universe=gene_universe,
            gmt=teacher.hallmark,
            prefix="MSIGDB_H",
            top_k=6,
        )
        reactome_hits = enrich_gmt(
            marker_genes=top_markers,
            gene_universe=gene_universe,
            gmt=teacher.reactome,
            prefix="REACTOME",
            top_k=6,
        )

        raw_tokens = teacher.layer_a_generate_tokens(
            cluster_name=f"{resolved_groupby}:{cluster}",
            top_markers=top_markers,
            go_hits=go_hits,
            hallmark_hits=hallmark_hits,
            reactome_hits=reactome_hits,
            max_tokens=8,
        )

        clean_tokens = teacher.layer_b_semantic_filtering(raw_tokens)
        constraints = teacher.layer_c_export_constraints(clean_tokens)

        all_results[cluster] = {
            "cluster": f"{resolved_groupby}:{cluster}",
            "top_markers": top_markers,
            "go_hits": [h.__dict__ for h in go_hits],
            "hallmark_hits": [h.__dict__ for h in hallmark_hits],
            "reactome_hits": [h.__dict__ for h in reactome_hits],
            "raw_tokens": [t.__dict__ for t in raw_tokens],
            "tokens": [t.__dict__ for t in clean_tokens],
            "constraints": constraints,
        }

    if output_name:
        json_name = output_name
    else:
        json_name = Path(h5ad_path).stem + "_tokens.json"

    out_json = os.path.join(out_dir, json_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return out_json
