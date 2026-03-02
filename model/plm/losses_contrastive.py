from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F


def cross_view_infonce(
    hs: torch.Tensor,
    ha: torch.Tensor,
    temperature: Union[float, torch.Tensor, None] = 0.2,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Symmetric InfoNCE between two views.

    hs: (B, d), ha: (B, d)

    If logit_scale is provided, logits are computed with CLIP-style scaling:
        logits = (hs @ ha.T) * exp(logit_scale)
    Otherwise, logits use fixed temperature:
        logits = (hs @ ha.T) / temperature
    """
    hs = F.normalize(hs, p=2, dim=-1)
    ha = F.normalize(ha, p=2, dim=-1)

    sim = hs @ ha.t()

    if logit_scale is not None:
        scale = torch.exp(logit_scale).clamp(min=eps, max=float(max_logit_scale))
        logits = sim * scale
    else:
        if temperature is None:
            temp = torch.tensor(0.2, device=hs.device, dtype=hs.dtype)
        elif isinstance(temperature, torch.Tensor):
            temp = temperature.to(device=hs.device, dtype=hs.dtype)
        else:
            temp = torch.tensor(float(temperature), device=hs.device, dtype=hs.dtype)
        temp = temp.clamp_min(eps)
        logits = sim / temp

    labels = torch.arange(hs.size(0), device=hs.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)
