"""
Evaluate a saved checkpoint and compare models.
Usage: python evaluate.py --model cnn_mlp --checkpoint checkpoints/cnn_mlp_best.pt
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from data.dataset import RadioSignalDataset
from models.cnn_mlp import CNNMLPModel


def load_model(name, checkpoint, cfg, device):
    if name == "cnn_mlp":
        model = CNNMLPModel(embed_dim=cfg.model.embed_dim, mlp_hidden=cfg.model.mlp_hidden)
    else:
        raise ValueError(name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, device):
    preds, targets = [], []
    for batch in loader:
        pred = model(batch["map"].to(device), batch["tx"].to(device), batch["rx"].to(device))
        preds.append(pred.cpu())
        targets.append(batch["label"])
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae  = np.mean(np.abs(preds - targets))
    return {"rmse_dBm": rmse, "mae_dBm": mae}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="cnn_mlp")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frequency",  default="415", choices=["415", "58"])
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = Config()

    all_scenes = list(range(cfg.data.n_scenes))
    rng = np.random.default_rng(cfg.train.seed)
    rng.shuffle(all_scenes)
    n_val = max(1, int(len(all_scenes) * cfg.data.val_fraction))
    val_scenes = all_scenes[:n_val]

    ds = RadioSignalDataset(
        data_root=cfg.data.data_root,
        frequency=args.frequency,
        scene_indices=val_scenes,
        n_tx_per_scene=cfg.data.n_tx_per_scene,
        rx_samples_per_tx=cfg.data.rx_samples_per_tx,
        nan_fill_db=cfg.data.nan_fill_db,
        seed=cfg.train.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)

    model   = load_model(args.model, args.checkpoint, cfg, device)
    metrics = evaluate(model, loader, device)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
