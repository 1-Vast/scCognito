from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SERSignals:
    token_vocab: list[str]
    c: torch.Tensor  # (N, K)
    token_texts: Optional[list[str]] = None


def _as_str_list(x: object) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out: list[str] = []
        for item in x:
            out.append("" if item is None else str(item))
        return out
    return None


def load_ser_signals(pt_path: str, device: str = "cuda") -> SERSignals:
    obj = torch.load(pt_path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"SER PT must be a dict, got type={type(obj)}")

    token_vocab = _as_str_list(obj.get("token_vocab")) or []
    c = obj["c"].to(device)

    token_texts: Optional[list[str]] = None
    token_texts = _as_str_list(obj.get("token_texts")) or _as_str_list(obj.get("token_names"))
    if token_texts is None and "token_meta" in obj:
        meta = obj.get("token_meta")
        if isinstance(meta, list) and meta and isinstance(meta[0], dict):
            token_texts = []
            for m in meta:
                name = m.get("name") or m.get("text") or m.get("label")
                if name is None:
                    name = m.get("id") or m.get("token") or ""
                token_texts.append(str(name))

    if token_texts is not None and len(token_texts) != len(token_vocab):
        token_texts = None

    return SERSignals(token_vocab=token_vocab, c=c, token_texts=token_texts)


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val not in {"0", "false", "no", "off"}


def _try_import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        return AutoTokenizer, AutoModel
    except Exception:
        return None, None


@torch.no_grad()
def _encode_texts(
    texts: Sequence[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 32,
    pooling: str = "mean",
) -> torch.Tensor:
    AutoTokenizer, AutoModel = _try_import_transformers()
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("transformers is required for text-driven prototype initialization.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_emb: list[torch.Tensor] = []
    pooling = str(pooling).strip().lower()
    if pooling not in {"mean", "cls"}:
        pooling = "mean"

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        last = out.last_hidden_state  # (B, L, H)

        if pooling == "cls":
            emb = last[:, 0, :]
        else:
            mask = inputs.get("attention_mask")
            if mask is None:
                emb = last.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).type_as(last)
                emb = (last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-6)

        all_emb.append(emb.detach().cpu())

    return torch.cat(all_emb, dim=0)


@torch.no_grad()
def _pca_reduce(x: torch.Tensor, d: int) -> torch.Tensor:
    if int(x.size(1)) == int(d):
        return x
    if int(d) > int(x.size(1)):
        pad = x.new_zeros(int(x.size(0)), int(d) - int(x.size(1)))
        return torch.cat([x, pad], dim=1)

    x0 = x - x.mean(dim=0, keepdim=True)
    q = min(int(d), int(x0.size(1)))
    _, _, V = torch.pca_lowrank(x0, q=q)
    return x0 @ V[:, : int(d)]


class TrainablePrototypes(nn.Module):
    """Trainable prototype vectors p_k in R^d."""

    def __init__(
        self,
        K: int,
        d: int,
        init_texts: Optional[Sequence[str]] = None,
        text_encoder_name: Optional[str] = None,
        text_batch_size: int = 16,
        text_max_length: int = 32,
        text_pooling: str = "mean",
        text_device: str = "cpu",
        enable_text_init: Optional[bool] = None,
    ):
        super().__init__()
        self.P = nn.Embedding(int(K), int(d))
        nn.init.normal_(self.P.weight, mean=0.0, std=0.02)

        if enable_text_init is None:
            enable_text_init = _env_flag("SC_COGNITO_TEXT_INIT", "1")
        if not bool(enable_text_init):
            return

        if init_texts is None:
            return

        texts = [str(t) for t in init_texts]
        if len(texts) != int(K):
            return

        if text_encoder_name is None:
            text_encoder_name = os.environ.get(
                "SC_COGNITO_TEXT_ENCODER",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ).strip()

        try:
            emb = _encode_texts(
                texts=texts,
                model_name=text_encoder_name,
                device=torch.device(text_device),
                batch_size=int(text_batch_size),
                max_length=int(text_max_length),
                pooling=text_pooling,
            )
            emb = F.normalize(emb, p=2, dim=-1)
            emb = _pca_reduce(emb, d=int(d))
            emb = F.normalize(emb, p=2, dim=-1).to(dtype=self.P.weight.dtype, device=self.P.weight.device)
            self.P.weight.data.copy_(emb)
            print(f"[OK][SER] prototypes initialized from text encoder: {text_encoder_name}", flush=True)
        except Exception as e:
            print(f"[WARN][SER] text-driven prototype init skipped: {type(e).__name__}: {e}", flush=True)

    def forward(self):
        return self.P.weight  # (K, d)


def _sample_negative_indices(
    pos_mask: torch.Tensor,  # (B, K) bool, True indicates positive tokens to exclude
    num_neg: int,
    max_tries: int = 8,
) -> torch.Tensor:
    if pos_mask.dtype != torch.bool:
        raise ValueError("pos_mask must be boolean.")

    B, K = int(pos_mask.size(0)), int(pos_mask.size(1))
    num_neg = int(min(max(0, num_neg), max(0, K)))
    if num_neg <= 0:
        return torch.empty((B, 0), device=pos_mask.device, dtype=torch.long)

    neg_idx = torch.randint(0, K, (B, num_neg), device=pos_mask.device, dtype=torch.long)
    if not torch.any(pos_mask):
        return neg_idx

    for _ in range(int(max_tries)):
        invalid = pos_mask.gather(1, neg_idx)
        if not bool(invalid.any()):
            break
        resample = torch.randint(0, K, (int(invalid.sum().item()),), device=pos_mask.device, dtype=torch.long)
        neg_idx[invalid] = resample
    return neg_idx


def semantic_energy(
    z: torch.Tensor,  # (N, d)
    c: torch.Tensor,  # (N, K)
    P: torch.Tensor,  # (K, d)
    w_proto: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,  # (N,), include in SER loss when True
    w_neg: float = 0.0,
    neg_samples: int = 64,
    neg_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Prototype alignment energy.

    Positive term:
        target_i = sum_k c[i,k] * p_k
        E_pos = 1 - cos(z_i, target_i)

    Negative term (optional):
        Sample prototypes with c[i,k] == 0 and penalize high cosine similarity:
        E_neg = mean(exp(cos(z_i, p_k) / t_neg))
    """
    if valid_mask is None:
        valid = torch.ones(int(z.size(0)), device=z.device, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=z.device)
        if valid.dtype != torch.bool:
            valid = valid > 0
        valid = valid.reshape(-1)
        if int(valid.numel()) != int(z.size(0)):
            raise ValueError(f"valid_mask size mismatch: got {int(valid.numel())} vs N={int(z.size(0))}")

    if not torch.any(valid):
        return torch.zeros((), device=z.device, dtype=z.dtype)

    target = c @ P
    e_pos = 1.0 - F.cosine_similarity(z, target, dim=1)
    loss_pos = float(w_proto) * e_pos[valid].mean()

    if float(w_neg) <= 0.0:
        return loss_pos

    z_v = z[valid]
    c_v = c[valid]
    pos_mask = (c_v > 0)

    neg_possible = (~pos_mask).any(dim=1)
    if not torch.any(neg_possible):
        return loss_pos

    z_n = z_v[neg_possible]
    pos_mask_n = pos_mask[neg_possible]

    neg_idx = _sample_negative_indices(pos_mask_n, num_neg=int(neg_samples))
    if int(neg_idx.numel()) == 0:
        return loss_pos

    P_neg = P[neg_idx]  # (B, C, d)
    sim_neg = F.cosine_similarity(z_n.unsqueeze(1), P_neg, dim=-1)  # (B, C)
    t_neg = float(max(1e-6, neg_temperature))
    e_neg = torch.exp(sim_neg / t_neg).mean()

    return loss_pos + float(w_neg) * e_neg
