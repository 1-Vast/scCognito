from __future__ import annotations

import json
import os
import sys

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
from .losses import (
    mask_gene_blocks,
    masked_recon_loss,
    spatial_neighbor_recon_loss,
    spatial_smoothness_loss,
)
from .losses_contrastive import cross_view_infonce
from .model import DualGraphEncoder, DecoderMLP
from .ser import load_ser_signals, TrainablePrototypes, semantic_energy


def _should_enable_tqdm() -> bool:
    flag = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _lam_ser_weight(cfg: PLMConfig, epoch_idx: int) -> float:
    warmup_ratio = float(max(0.0, min(1.0, cfg.lam_ser_warmup_ratio)))
    warmup_epochs = int(round(float(cfg.epochs) * warmup_ratio))
    if warmup_epochs <= 0:
        return float(cfg.lam_ser)
    if epoch_idx >= warmup_epochs:
        return float(cfg.lam_ser)
    return float(cfg.lam_ser) * (float(epoch_idx) / float(max(1, warmup_epochs)))


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
    if int(c.size(0)) != int(x.size(0)):
        raise ValueError(
            f"SER c has N={int(c.size(0))} but data has N={int(x.size(0))}. "
            f"Bridge must be from the same dataset."
        )

    coverage_mask = (c.sum(dim=1) > 0)
    covered_n = int(coverage_mask.sum().item())
    total_n = int(coverage_mask.numel())
    if covered_n <= 0:
        print("[WARN][SER] Zero coverage. SER disabled.", flush=True)
    else:
        print(f"[INFO][SER] Coverage cells={covered_n}/{total_n}", flush=True)

    encoder = DualGraphEncoder(
        d_in=int(x.size(1)),
        d_hid=int(cfg.d_hid),
        d_out=int(cfg.d_out),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        global_attn=bool(cfg.global_attn),
        global_attn_heads=int(cfg.global_attn_heads),
        global_attn_chunk_q=int(cfg.global_attn_chunk_q),
        global_attn_max_n=int(cfg.global_attn_max_n),
        global_attn_dropout=float(cfg.global_attn_dropout),
        global_attn_topk=int(getattr(cfg, "global_attn_topk", 128)),
        contrastive_init_temp=float(cfg.contrast_temp),
        learnable_contrastive_temp=True,
    ).to(device)

    decoder = DecoderMLP(d_z=int(cfg.d_out), d_x=int(x.size(1))).to(device)

    K = len(ser.token_vocab)
    token_texts = ser.token_texts if ser.token_texts is not None else ser.token_vocab
    prototypes = TrainablePrototypes(
        K=K,
        d=int(cfg.d_out),
        init_texts=token_texts,
        text_encoder_name=getattr(cfg, "text_encoder_name", None),
        text_batch_size=int(getattr(cfg, "text_batch_size", 16)),
        text_max_length=int(getattr(cfg, "text_max_length", 32)),
        text_pooling=str(getattr(cfg, "text_pooling", "mean")),
        text_device=str(getattr(cfg, "text_device", "cpu")),
        enable_text_init=getattr(cfg, "enable_text_init", None),
    ).to(device)

    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(prototypes.parameters())
    opt = AdamW(all_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    if int(cfg.epochs) < 500:
        print(
            f"[WARN][TRAIN] Epochs limited ({int(cfg.epochs)}). "
            "Consider 500-2000 epochs.",
            flush=True,
        )

    print(f"[PROGRESS][PLM] epoch=0/{int(cfg.epochs)} pct=0.0", flush=True)
    
    pbar = tqdm(
        range(1, int(cfg.epochs) + 1),
        total=int(cfg.epochs),
        desc="PLM Train",
        unit="ep",
        ascii=True,
        ncols=140,
        colour="green",
        disable=(not _should_enable_tqdm()),
    )

    device_type = device.type
    use_amp = (device_type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
    mode = str(getattr(cfg, "mode", "finetune") or "finetune").strip().lower()
    if mode not in {"pretrain", "finetune"}:
        mode = "finetune"

    ser_w_neg = float(getattr(cfg, "ser_w_neg", 0.1))
    ser_neg_samples = int(getattr(cfg, "ser_neg_samples", 64))
    ser_neg_temp = float(getattr(cfg, "ser_neg_temp", 1.0))

    for ep in pbar:
        encoder.train()
        decoder.train()
        prototypes.train()
        opt.zero_grad(set_to_none=True)

        x_m, mask = mask_gene_blocks(x, float(cfg.mask_ratio))

        with torch.autocast(device_type=device_type, enabled=use_amp):
            hs, ha = encoder.encode_streams(x_m, edge_s, edge_a)
            z = encoder.fuse_streams(hs, ha)
            x_hat = decoder(z)

            l_recon = masked_recon_loss(x_true=x, x_pred=x_hat, mask=mask)
            l_spatial = spatial_neighbor_recon_loss(
                x_hat_center=x_hat,
                x_true=x,
                mask=mask,
                edge_spatial=edge_s,
                max_edges=int(cfg.spatial_max_edges),
            )
            l_smooth = spatial_smoothness_loss(z, edge_s)

        lam_ser_now = 0.0
        e_sem = torch.zeros((), device=device, dtype=z.dtype)
        if mode == "finetune" and covered_n > 0 and float(cfg.lam_ser) > 0.0:
            P = prototypes()
            e_sem = semantic_energy(
                z=z,
                c=c,
                P=P,
                w_proto=float(cfg.ser_w_proto),
                valid_mask=coverage_mask,
                w_neg=ser_w_neg,
                neg_samples=ser_neg_samples,
                neg_temperature=ser_neg_temp,
            )
            lam_ser_now = _lam_ser_weight(cfg, ep)

        l_contrast = torch.zeros((), device=device, dtype=z.dtype)
        if ha is not None and float(cfg.w_contrast) > 0.0:
            l_contrast = cross_view_infonce(hs, ha, logit_scale=encoder.logit_scale)

        loss = (
            float(cfg.w_recon) * l_recon
            + float(cfg.w_spatial_pred) * l_spatial
            + float(getattr(cfg, "w_spatial_smooth", 0.0)) * l_smooth
            + float(lam_ser_now) * e_sem
            + float(cfg.w_contrast) * l_contrast
        )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, float(cfg.grad_clip))
            opt.step()

        actual_log_every = max(int(cfg.log_every), int(cfg.epochs) // 20)
        if actual_log_every == 0: 
            actual_log_every = 1

        if ep % actual_log_every == 0 or ep == 1 or ep == int(cfg.epochs):
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                recon=f"{l_recon.item():.3f}",
                smooth=f"{l_smooth.item():.3f}",
                ctr=f"{l_contrast.item():.3f}",
                ser=f"{e_sem.item():.3f}"
            )
            
            pct = 100.0 * float(ep) / float(max(1, int(cfg.epochs)))
            
            msg = (
                f"[PROGRESS][PLM] ep={ep}/{int(cfg.epochs)} pct={pct:.1f}% | "
                f"loss={loss.item():.4f} recon={l_recon.item():.4f} "
                f"smooth={l_smooth.item():.4f} ctr={l_contrast.item():.4f} "
                f"SER={e_sem.item():.4f}"
            )
            
            if getattr(pbar, "disable", False):
                print(msg, flush=True)
            else:
                tqdm.write(msg)

    encoder.eval()
    decoder.eval()
    prototypes.eval()
    with torch.no_grad():
        z_stable = encoder(x, edge_s, edge_a)
        x_hat_stable = decoder(z_stable)

        recon_cell = F.mse_loss(x_hat_stable, x, reduction="none").mean(dim=1).cpu()

        if covered_n > 0:
            P_stable = prototypes()
            target_stable = c @ P_stable
            ser_cell = 1.0 - torch.cosine_similarity(z_stable, target_stable, dim=1)
            ser_cell = ser_cell.masked_fill(~coverage_mask, float("nan")).cpu()
        else:
            ser_cell = torch.full((int(x.size(0)),), float("nan"), dtype=torch.float32).cpu()

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

    finite = torch.isfinite(ser_cell)
    if bool(finite.any()):
        ser_mean = float(ser_cell[finite].mean().item())
    else:
        ser_mean = 0.0

    train_report = {
        "version": 1,
        "stability": {
            "final_loss": float(loss.item()),
            "recon_loss": float(l_recon.item()),
            "spatial_neighbor_loss": float(l_spatial.item()),
            "spatial_smooth_loss": float(l_smooth.item()),
        },
        "semantic_alignment": {
            "ser_energy_mean": ser_mean,
            "ser_coverage": f"{covered_n}/{total_n}",
            "mode": mode,
        },
        "config_used": {
            "mode": mode,
            "epochs": int(cfg.epochs),
            "mask_ratio": float(cfg.mask_ratio),
            "lam_ser_final": float(lam_ser_now),
            "w_spatial_smooth": float(getattr(cfg, "w_spatial_smooth", 0.0)),
            "w_contrast": float(getattr(cfg, "w_contrast", 0.0)),
            "d_hid": int(cfg.d_hid),
            "d_out": int(cfg.d_out),
            "n_layers": int(cfg.n_layers),
        },
    }
    report_path = cfg.out_dir / "train_report.json"
    report_path.write_text(json.dumps(train_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] saved:", report_path, flush=True)

    return out_ckpt