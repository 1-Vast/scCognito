import torch
import torch.nn.functional as F

def cross_view_infonce(hs: torch.Tensor, ha: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Symmetric InfoNCE between two views:
    hs: (B, d), ha: (B, d)
    """
    hs = F.normalize(hs, p=2, dim=-1)
    ha = F.normalize(ha, p=2, dim=-1)

    logits = (hs @ ha.t()) / max(1e-6, float(temperature))  # (B, B)
    labels = torch.arange(hs.size(0), device=hs.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)
