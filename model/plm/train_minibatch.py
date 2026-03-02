import torch
from torch.optim import AdamW

def run_train_minibatch(cfg: PLMConfig):
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    batch = load_plm_batch(...)
    x_all = batch.x
    edge_s = batch.edge_spatial

    data = Data(x=x_all, edge_index=edge_s)
    loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],   # 2-hop sampling
        batch_size=1024,
        shuffle=True,
        input_nodes=None,
    )

    encoder = DualGraphEncoder(...).to(device)
    decoder = DecoderMLP(...).to(device)
    opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        decoder.train()
        for sub in loader:
            sub = sub.to(device)
            x = sub.x
            edge = sub.edge_index
            x_m, mask = mask_gene_blocks(x, cfg.mask_ratio)
            z = encoder(x_m, edge, edge_attr=None)
            x_hat = decoder(z)
            l_recon = masked_recon_loss(x, x_hat, mask)

            opt.zero_grad(set_to_none=True)
            l_recon.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), cfg.grad_clip)
            opt.step()
