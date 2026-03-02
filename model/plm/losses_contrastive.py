from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F


def cross_view_infonce(
    hs: torch.Tensor,
    ha: torch.Tensor,
    temperature: Union[float, torch.Tensor, None] = 0.2,
    logit_scale: Optional[torch.Tensor] = None,
    scale: Optional[Union[float, torch.Tensor]] = None,
    max_logit_scale: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Symmetric InfoNCE between two views.

    hs: (B, d), ha: (B, d)

    Priority:
      1) If `scale` is provided, logits = sim * clamp(scale).
      2) Else if `logit_scale` is provided, logits = sim * clamp(exp(logit_scale)).
      3) Else logits = sim / clamp(temperature).
    """
    hs = F.normalize(hs, p=2, dim=-1)
    ha = F.normalize(ha, p=2, dim=-1)

    sim = hs @ ha.t()

    if scale is not None:
        if isinstance(scale, torch.Tensor):
            s = scale.to(device=hs.device, dtype=hs.dtype)
        else:
            s = torch.tensor(float(scale), device=hs.device, dtype=hs.dtype)
        s = s.clamp(min=eps, max=float(max_logit_scale))
        logits = sim * s
    elif logit_scale is not None:
        s = torch.exp(logit_scale).clamp(min=eps, max=float(max_logit_scale))
        logits = sim * s
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
