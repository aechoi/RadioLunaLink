import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import parse_args
from data.dataset import RadioSignalDataset
from models.cnn_mlp import CNNMLPModel
from models.vit_attention import CoordCondViT


def build_model(cfg):
    if cfg.model.name == "cnn_mlp":
        return CNNMLPModel(embed_dim=cfg.model.embed_dim, mlp_hidden=cfg.model.mlp_hidden)
    if cfg.model.name == "vit":
        return CoordCondViT(embed_dim=cfg.model.embed_dim, mlp_hidden=cfg.model.mlp_hidden,
                            n_heads=cfg.model.vit_heads, n_layers=cfg.model.vit_layers,
                            patch_size=cfg.model.vit_patch_size)
    raise ValueError(f"Unknown model: {cfg.model.name}")


def run_epoch(model, loader, optimizer, device, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    total_sq_err, n = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc=desc, leave=False):
            hm    = batch["map"].to(device)
            tx    = batch["tx"].to(device)
            rx    = batch["rx"].to(device)
            label = batch["label"].to(device)
            pred  = model(hm, tx, rx)
            loss  = nn.functional.mse_loss(pred, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_sq_err += loss.item() * label.numel()
            n += label.numel()
    return (total_sq_err / n) ** 0.5  # RMSE


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {cfg.model.name}")

    all_scenes = list(range(cfg.data.n_scenes))
    rng = np.random.default_rng(cfg.train.seed)
    rng.shuffle(all_scenes)
    n_val = max(1, int(len(all_scenes) * cfg.data.val_fraction))
    val_scenes, train_scenes = all_scenes[:n_val], all_scenes[n_val:]

    ds_kwargs = dict(
        data_root=cfg.data.data_root,
        frequency=cfg.data.frequency,
        n_tx_per_scene=cfg.data.n_tx_per_scene,
        rx_samples_per_tx=cfg.data.rx_samples_per_tx,
        nan_fill_db=cfg.data.nan_fill_db,
        seed=cfg.train.seed,
    )
    print("building datasets...")
    train_ds = RadioSignalDataset(scene_indices=train_scenes, **ds_kwargs)
    val_ds   = RadioSignalDataset(scene_indices=val_scenes,   **ds_kwargs)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,} samples")

    print("building dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    print("building model...")
    model     = build_model(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    print("starting training...\n")
    best_val_rmse = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_rmse = run_epoch(model, train_loader, optimizer, device, desc=f"epoch {epoch} train")
        val_rmse   = run_epoch(model, val_loader,   None,      device, desc=f"epoch {epoch} val  ")
        print(f"epoch {epoch:3d}  train_rmse={train_rmse:.3f}  val_rmse={val_rmse:.3f} dBm")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), ckpt_dir / f"{cfg.model.name}_best.pt")
            print(f"          -> saved checkpoint (val_rmse={best_val_rmse:.3f})")


if __name__ == "__main__":
    main()
