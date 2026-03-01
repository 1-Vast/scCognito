from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from openai import OpenAI

from dotenv import load_dotenv
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
load_dotenv(_THIS_DIR / ".env.teacher", override=True)

# ----------------------------
# Configs (keep small and explicit)
# ----------------------------

@dataclass
class TeacherConfig:
    # Layer A: generation
    vote_runs: int = 5
    temperature: float = 0.25
    max_tokens: int = 12

    # Layer B: filtering
    min_confidence: float = 0.20
    min_evidence_items: int = 1
    allow_literature_only: bool = False

    # Layer C: shaping compilation
    # (C is "how tokens become learnable operators"; here we only compile signals, not train)
    semantic_edge_only: bool = True


# ----------------------------
# Utilities
# ----------------------------

def _extract_output_text(response) -> str:
    """Extract concatenated assistant output_text from Responses API."""
    full_text = ""
    for item in response.output:
        if item.type == "message" and item.role == "assistant":
            for c in item.content:
                if c.type == "output_text":
                    full_text += c.text
    return full_text.strip()


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Parse JSON strictly; raise ValueError with helpful info if invalid."""
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON. Raw:\n{s}\nError: {e}") from e


# ----------------------------
# Main: Semantic Teacher (Layer A + B + C in one file)
# ----------------------------

class SemanticTeacher:
    """
      - Layer A: Token Generation (LLM)
      - Layer B: Semantic Filtering (constraints, evidence gating, granularity fallback)
      - Layer C: Representation Shaping signals compilation (c/p/S_semantic)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        cfg: Optional[TeacherConfig] = None,
        ontology_ids: Optional[set[str]] = None,
        mutex_pairs: Optional[set[Tuple[str, str]]] = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.cfg = cfg or TeacherConfig()

        # Ontology IDs allowlist. If None/empty, ontology check is bypassed.
        self.ontology_ids = ontology_ids or set()

        # Hard mutual exclusion constraints.
        self.mutex_pairs = mutex_pairs or set()

        # System prompt enforces JSON-only output with grounded evidence.
        self.system_prompt = f"""You are a biomedical semantic teacher.
Return ONLY valid JSON (no markdown, no extra text).

Task: Generate semantic tokens grounded in provided evidence.

Rules:
1) Every token MUST include evidence items. Evidence refs must use the provided ref strings.
2) Confidence must be in [0,1]. Do not output tokens with confidence < 0.05.
3) If candidate_token_ids are provided, do NOT invent ontology IDs.
4) Output must follow the schema exactly.

Schema:
{{
  "tokens":[
    {{"type":"GO|Pathway|CellState|Phenotype|Regulation",
      "id":"string",
      "name":"string",
      "confidence":0.0,
      "evidence":[{{"source":"marker|enrichment|literature","ref":"string","weight":1.0}}]
    }}
  ],
  "unknown_flag": false,
  "notes": "string"
}}

Max tokens per run: {self.cfg.max_tokens}
"""

    # ============================
    # Public API
    # ============================

    def run(
        self,
        markers: List[Dict[str, Any]],
        enrichment: Optional[List[Dict[str, Any]]] = None,
        candidate_token_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        cluster_ids: Optional[List[Any]] = None,
        adjacency: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        主入口：给一个 anchor（markers/enrichment/...）输出 tokens +（可选）Layer C 编译产物。
        - If you only want Layer A/B, do not pass cluster_ids/adjacency/embeddings.
        """

        # Layer A: token generation (with voting)
        merged = self._layerA_generate(markers, enrichment, candidate_token_ids, context)

        # Layer B: semantic filtering
        filtered = self._layerB_filter(merged)

        out = {
            "tokens": filtered["tokens"],
            "unknown_flag": filtered.get("unknown_flag", False),
            "notes": filtered.get("notes", ""),
        }

        # Layer C: compile shaping signals if requested
        if cluster_ids is not None and embeddings is not None:
            compiled = self._layerC_compile(cluster_ids, filtered["tokens"], embeddings, adjacency)
            out.update(compiled)

        return out

    # ============================
    # Layer A: Token Generation
    # ============================

    def _layerA_generate(
        self,
        markers: List[Dict[str, Any]],
        enrichment: Optional[List[Dict[str, Any]]],
        candidate_token_ids: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Layer A: call LLM multiple times and merge by self-consistency voting."""

        payload = self._build_anchor_payload(markers, enrichment, candidate_token_ids, context)

        votes = []
        for k in range(self.cfg.vote_runs):
            raw = self._call_llm_json(payload, temperature=self.cfg.temperature)
            votes.append(raw)
            time.sleep(0.05)

        merged = self._merge_votes(votes)
        return merged

    def _build_anchor_payload(
        self,
        markers: List[Dict[str, Any]],
        enrichment: Optional[List[Dict[str, Any]]],
        candidate_token_ids: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a compact, verifiable evidence payload."""
        m_refs = [f"marker:{m['gene']}" for m in markers if "gene" in m]
        e_refs = []
        if enrichment:
            for e in enrichment:
                if "id" in e:
                    e_refs.append(f"enrich:{e['id']}")
                elif "name" in e:
                    e_refs.append(f"enrich:{e['name']}")

        return {
            "evidence": {
                "markers": markers,
                "enrichment": enrichment or [],
                "allowed_marker_refs": m_refs,
                "allowed_enrich_refs": e_refs,
            },
            "constraints": {
                "max_tokens": self.cfg.max_tokens,
                "candidate_token_ids": candidate_token_ids or [],
            },
            "context": context or {},
        }

    def _call_llm_json(self, user_payload: Dict[str, Any], temperature: float = 0.2) -> Dict[str, Any]:
        """Call standard OpenAI-compatible completions API and require JSON."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=temperature,
        )
        text = resp.choices[0].message.content
        return _safe_json_loads(text)

    def _merge_votes(self, votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple LLM outputs.
        - Token confidence is frequency across votes.
        - Evidence is unioned by ref.
        """
        V = max(1, len(votes))
        unknown_rate = sum(1 for v in votes if v.get("unknown_flag", False)) / V

        freq = Counter()
        meta = defaultdict(lambda: {"type": None, "id": None, "name": None, "evidence": {}})

        for v in votes:
            for t in v.get("tokens", []):
                typ = t.get("type", "UNK")
                tid = t.get("id", "")
                name = t.get("name", "")
                key = (typ, tid, name)
                freq[key] += 1

                m = meta[key]
                m["type"], m["id"], m["name"] = typ, tid, name

                for e in t.get("evidence", []):
                    ref = e.get("ref", "")
                    if not ref:
                        continue
                    m["evidence"][ref] = {
                        "source": e.get("source", "marker"),
                        "ref": ref,
                        "weight": float(e.get("weight", 1.0)),
                    }

        tokens = []
        for (typ, tid, name), c in freq.items():
            conf = c / V
            ev = list(meta[(typ, tid, name)]["evidence"].values())
            tokens.append({
                "type": typ,
                "id": tid,
                "name": name,
                "confidence": float(conf),
                "evidence": ev,
            })

        tokens.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "tokens": tokens,
            "unknown_flag": bool(unknown_rate > 0.5 and len(tokens) == 0),
            "notes": f"unknown_rate={unknown_rate:.3f}",
        }

    # ============================
    # Layer B: Semantic Filtering
    # ============================

    def _layerB_filter(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Layer B: ontology check, mutual exclusion, evidence gating, granularity fallback."""
        toks = sorted(out.get("tokens", []), key=lambda x: x.get("confidence", 0.0), reverse=True)

        kept = []
        chosen = set()

        for t in toks:
            if not self._evidence_ok(t):
                continue

            # Ontology validity check (bypass if no ontology loaded)
            if self.ontology_ids and (t.get("id") not in self.ontology_ids):
                t2 = self._granularity_fallback(t)
                if t2 is None or not self._evidence_ok(t2):
                    continue
                t = t2

            # Mutual exclusion
            if self._violates_mutex(chosen, t.get("id", "")):
                continue

            kept.append(t)
            chosen.add(t.get("id", ""))

        return {
            "tokens": kept,
            "unknown_flag": bool(out.get("unknown_flag", False) and len(kept) == 0),
            "notes": out.get("notes", ""),
        }

    def _evidence_ok(self, t: Dict[str, Any]) -> bool:
        """Evidence gating and confidence thresholding."""
        conf = float(t.get("confidence", 0.0))
        if conf < self.cfg.min_confidence:
            return False

        ev = t.get("evidence", []) or []
        if len(ev) < self.cfg.min_evidence_items:
            return False

        if not self.cfg.allow_literature_only:
            for e in ev:
                if e.get("source") in ("marker", "enrichment"):
                    return True
            return False

        return True

    def _violates_mutex(self, chosen: set, token_id: str) -> bool:
        """Hard mutex constraints."""
        for a, b in self.mutex_pairs:
            if token_id == a and b in chosen:
                return True
            if token_id == b and a in chosen:
                return True
        return False

    def _granularity_fallback(self, t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Granularity downgrade placeholder.
        Replace with ontology-parent lookup later if you load GO DAG.
        """
        typ = t.get("type", "")
        if typ in ("GO", "Pathway"):
            return {
                "type": typ,
                "id": f"{typ}:COARSE",
                "name": f"{typ} (coarse)",
                "confidence": max(0.10, min(0.30, float(t.get("confidence", 0.0)))),
                "evidence": t.get("evidence", []),
            }
        if typ in ("CellState", "Phenotype", "Regulation"):
            return {
                "type": "CellState",
                "id": "CellState:COARSE",
                "name": "Cell state (coarse)",
                "confidence": max(0.10, min(0.30, float(t.get("confidence", 0.0)))),
                "evidence": t.get("evidence", []),
            }
        return None

    # ============================
    # Layer C: Representation Shaping (compile learnable operators)
    # ============================

    def _layerC_compile(
        self,
        cluster_ids: List[Any],
        tokens: List[Dict[str, Any]],
        embeddings: Any,
        adjacency: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Layer C compiles:
          - token_vocab: fixed K tokens
          - cluster_token_conf: per-cluster token distribution
          - (optional) semantic affinity matrix S_semantic (masked by adjacency)
        NOTE: This does not perform training; it only builds signals.
        """
        # 1) token vocab
        vocab = [t["id"] for t in tokens]
        token_index = {tid: i for i, tid in enumerate(vocab)}

        # 2) cluster-level distribution (here: single cluster case or provided externally)
        # If you have multiple clusters, you should call run() per cluster and merge outside.
        # Here we keep minimal: broadcast same tokens to all clusters present.
        uniq = sorted({str(x) for x in cluster_ids})
        cluster_token_conf = {cid: {tid: float(tokens[i]["confidence"]) for tid, i in token_index.items()} for cid in uniq}

        # 3) cell-level c matrix
        import numpy as np
        import torch

        N = len(cluster_ids)
        K = len(vocab)
        c = torch.zeros((N, K), dtype=torch.float32)

        for i, g in enumerate(cluster_ids):
            conf = cluster_token_conf.get(str(g), {})
            for tid, val in conf.items():
                j = token_index.get(tid, None)
                if j is not None:
                    c[i, j] = float(val)

        c = c / (c.sum(dim=1, keepdim=True) + 1e-8)

        # 4) prototype p (weighted mean)
        z = embeddings
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(np.asarray(z), dtype=torch.float32)
        p = (c.T @ z) / (c.sum(dim=0).unsqueeze(1) + 1e-12)

        # 5) semantic affinity (optional)
        S_sem = None
        if adjacency is not None:
            A = adjacency
            if not isinstance(A, torch.Tensor):
                A = torch.tensor(np.asarray(A), dtype=torch.float32)

            c_norm = c / (torch.norm(c, dim=1, keepdim=True) + 1e-12)
            S = c_norm @ c_norm.T
            if self.cfg.semantic_edge_only:
                S = S * (A > 0).float()
            S_sem = S

        return {
            "token_vocab": vocab,
            "c": c,
            "p": p,
            "S_semantic": S_sem,
            "cluster_token_conf": cluster_token_conf,
        }


if __name__ == "__main__":
    teacher = SemanticTeacher(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=os.getenv("ARK_API_KEY"),
        model="ep-20260227163508-kxp4m" 
    )

    out = teacher.run(
        markers=[
            {"gene": "IL7R"}, 
            {"gene": "CCR7"}
        ],
        enrichment=[
            {"id": "GO:0006955", "name": "immune response"}
        ]
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))