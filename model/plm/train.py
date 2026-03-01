from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

from .config import PLMConfig
from .data import load_plm_batch
from .model import DualGraphEncoder, DecoderMLP
from .losses import (
    mask_gene_blocks,
    masked_recon_loss,
    spatial_neighbor_recon_loss,
)
from .ser import load_ser_signals, TrainablePrototypes, semantic_energy


def run_train(cfg: PLMConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    batch = load_plm_batch(
        h5ad_path=str(cfg.h5ad_path),
        use_rep=cfg.use_rep,
        pca_dim=cfg.pca_dim,
        use_hvg=cfg.use_hvg,
        hvg_top=cfg.hvg_top,
        spatial_graph=cfg.spatial_graph,
        spatial_k=cfg.spatial_k,
        spatial_radius=cfg.spatial_radius,
        attribute_graph=cfg.attribute_graph,
        attr_k=cfg.attr_k,
    )

    x = batch.x.to(device)
    edge_s = batch.edge_spatial.to(device)
    edge_a = batch.edge_attr.to(device) if batch.edge_attr is not None else None

    ser = load_ser_signals(str(cfg.ser_pt_path), device=str(device))
    c = ser.c
    if c.size(0) != x.size(0):
        raise ValueError(
            f"SER c has N={c.size(0)} but data has N={x.size(0)}. "
            f"Make sure the bridge was built from the same dataset."
        )

    encoder = DualGraphEncoder(
        d_in=x.size(1),
        d_hid=cfg.d_hid,
        d_out=cfg.d_out,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    decoder = DecoderMLP(d_z=cfg.d_out, d_x=x.size(1)).to(device)

    K = len(ser.token_vocab)
    prototypes = TrainablePrototypes(K=K, d=cfg.d_out).to(device)

    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(prototypes.parameters())
    opt = AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"[PROGRESS][PLM] epoch=0/{cfg.epochs} pct=0.0", flush=True)
    pbar = tqdm(range(1, cfg.epochs + 1), total=cfg.epochs, desc="PLM Train", unit="ep")
    for ep in pbar:
        encoder.train()
        decoder.train()
        prototypes.train()

        x_m, mask = mask_gene_blocks(x, cfg.mask_ratio)

        z = encoder(x_m, edge_s, edge_a)
        x_hat = decoder(z)

        l_recon = masked_recon_loss(x, x_hat, mask)
        l_spatial = spatial_neighbor_recon_loss(
            z=z,
            decoder=decoder,
            x_true=x,
            mask=mask,
            edge_spatial=edge_s,
            max_edges=cfg.spatial_max_edges,
        )

        P = prototypes()
        e_sem = semantic_energy(z=z, c=c, P=P, w_proto=cfg.ser_w_proto)

        loss = cfg.w_recon * l_recon + cfg.w_spatial_pred * l_spatial + cfg.lam_ser * e_sem

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        opt.step()

        if ep % cfg.log_every == 0 or ep == 1 or ep == cfg.epochs:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{l_recon.item():.4f}",
                ser=f"{e_sem.item():.4f}",
            )
            pct = 100.0 * float(ep) / float(max(1, cfg.epochs))
            print(
                f"[PROGRESS][PLM] epoch={ep}/{cfg.epochs} pct={pct:.1f} "
                f"loss={loss.item():.4f} recon={l_recon.item():.4f} "
                f"spNBR={l_spatial.item():.4f} SER={e_sem.item():.4f}",
                flush=True,
            )

    # -----------------------------------------------------------------
    # Post-training deterministic evaluation (stable metrics for agent)
    # -----------------------------------------------------------------
    encoder.eval()
    decoder.eval()
    prototypes.eval()
    with torch.no_grad():
        z_stable = encoder(x, edge_s, edge_a)
        x_hat_stable = decoder(z_stable)

        recon_cell = F.mse_loss(x_hat_stable, x, reduction="none").mean(dim=1).cpu()

        P_stable = prototypes()
        target_stable = c @ P_stable
        ser_cell = (1.0 - torch.cosine_similarity(z_stable, target_stable, dim=1)).cpu()

        last_metrics = {
            "recon_err_cell": recon_cell.numpy(),
            "ser_energy_cell": ser_cell.numpy(),
        }

    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "prototypes": prototypes.state_dict(),
        "token_vocab": ser.token_vocab,
        "config": cfg.__dict__,
        "last_metrics": last_metrics,
    }

    out_ckpt = cfg.out_dir / "plm_ckpt.pt"
    torch.save(ckpt, out_ckpt)
    print("[OK] saved:", out_ckpt)

    return out_ckpt
