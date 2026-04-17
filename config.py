from dataclasses import dataclass, field
from pathlib import Path
import argparse

_DATA_DEFAULT = str(
    Path(__file__).parent.parent.parent
    / "NASA_DCGR_NETWORKING/radio_data_2/radio_data_2"
)


@dataclass
class DataConfig:
    data_root: str = _DATA_DEFAULT
    frequency: str = "415"       # "415" or "58"
    n_scenes: int = 500
    n_tx_per_scene: int = 50
    rx_samples_per_tx: int = 32  # random RX positions sampled per (scene, TX) pair
    nan_fill_db: float = -200.0  # sentinel for no-signal (NaN) pixels
    val_fraction: float = 0.1


@dataclass
class ModelConfig:
    name: str = "cnn_mlp"        # "cnn_mlp" | "vit"
    embed_dim: int = 256
    mlp_hidden: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50       # batches


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=cfg.data.data_root)
    p.add_argument("--frequency", default=cfg.data.frequency, choices=["415", "58"])
    p.add_argument("--n_scenes", type=int, default=cfg.data.n_scenes)
    p.add_argument("--n_tx_per_scene", type=int, default=cfg.data.n_tx_per_scene)
    p.add_argument("--rx_samples_per_tx", type=int, default=cfg.data.rx_samples_per_tx)
    p.add_argument("--model", default=cfg.model.name, choices=["cnn_mlp", "vit"])
    p.add_argument("--embed_dim", type=int, default=cfg.model.embed_dim)
    p.add_argument("--mlp_hidden", type=int, default=cfg.model.mlp_hidden)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--lr", type=float, default=cfg.train.lr)
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--checkpoint_dir", default=cfg.train.checkpoint_dir)
    args = p.parse_args()

    cfg.data.data_root = args.data_root
    cfg.data.frequency = args.frequency
    cfg.data.n_scenes = args.n_scenes
    cfg.data.n_tx_per_scene = args.n_tx_per_scene
    cfg.data.rx_samples_per_tx = args.rx_samples_per_tx
    cfg.model.name = args.model
    cfg.model.embed_dim = args.embed_dim
    cfg.model.mlp_hidden = args.mlp_hidden
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    cfg.train.checkpoint_dir = args.checkpoint_dir
    return cfg
