import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class RadioSignalDataset(Dataset):
    """
    Each sample: (heightmap, tx_coord, rx_coord) -> signal_strength_dBm.

    Everything is pre-loaded at init so __getitem__ is pure array indexing
    with no disk I/O during training.
    """

    def __init__(
        self,
        data_root,
        frequency="415",
        scene_indices=None,
        n_tx_per_scene=50,
        rx_samples_per_tx=32,
        nan_fill_db=-200.0,
        seed=42,
    ):
        self.root = Path(data_root)
        self.rm_dir = f"rm{frequency}"
        self.nan_fill = np.float32(nan_fill_db)

        # Infer map size from first heightmap
        hm0 = np.load(self.root / "hm" / "hm_0.npy")
        self.H, self.W = hm0.shape[:2]

        # Pre-sample RX pixel coordinates reproducibly
        rng = np.random.default_rng(seed)
        # rx_coords[scene][tx_idx] = list of (row, col) tuples
        rx_coords: dict[int, dict[int, list]] = {}
        self.index = []  # (scene_idx, tx_idx, rx_row, rx_col)
        for i in scene_indices:
            rx_coords[i] = {}
            for j in range(n_tx_per_scene):
                rows = rng.integers(0, self.H, size=rx_samples_per_tx)
                cols = rng.integers(0, self.W, size=rx_samples_per_tx)
                pairs = list(zip(rows.tolist(), cols.tolist()))
                rx_coords[i][j] = pairs
                for r, c in pairs:
                    self.index.append((i, j, r, c))

        # ── Pre-load heightmaps (one per scene) ──────────────────────────────
        print("  caching heightmaps...")
        self.hm_cache: dict[int, np.ndarray] = {}
        for i in tqdm(scene_indices, leave=False):
            hm = np.load(self.root / "hm" / f"hm_{i}.npy").astype(np.float32)
            self.hm_cache[i] = (hm - hm.mean()) / (hm.std() + 1e-6)

        # ── Pre-extract TX coords and labels (one rm+tx load per (scene,tx)) ─
        print("  extracting TX coords and labels...")
        self.tx_coords: dict[tuple, np.ndarray] = {}
        self.labels = np.empty(len(self.index), dtype=np.float32)
        label_idx = 0
        for i in tqdm(scene_indices, leave=False):
            for j in range(n_tx_per_scene):
                tx_map = np.load(self.root / "tx" / f"tx_{i}_{j}.npy")
                tx_loc = np.argwhere(tx_map != 0)
                self.tx_coords[(i, j)] = np.array(
                    [tx_loc[0, 1] / self.W, tx_loc[0, 0] / self.H], dtype=np.float32
                )

                rm = np.load(
                    self.root / self.rm_dir / f"rm_{i}_{j}.npy"
                ).astype(np.float32)
                for ry, rx in rx_coords[i][j]:
                    val = rm[ry, rx]
                    self.labels[label_idx] = self.nan_fill if np.isnan(val) else val
                    label_idx += 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene, tx_idx, ry, rx = self.index[idx]
        hm       = self.hm_cache[scene][np.newaxis]                        # (1, H, W)
        tx_coord = self.tx_coords[(scene, tx_idx)]                         # (2,)
        rx_coord = np.array([rx / self.W, ry / self.H], dtype=np.float32) # (2,)
        return {
            "map":   torch.from_numpy(hm),
            "tx":    torch.from_numpy(tx_coord),
            "rx":    torch.from_numpy(rx_coord),
            "label": torch.tensor(self.labels[idx]),
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import Config

    cfg = Config()
    ds = RadioSignalDataset(
        data_root=cfg.data.data_root,
        frequency=cfg.data.frequency,
        scene_indices=list(range(5)),
        n_tx_per_scene=2,
        rx_samples_per_tx=4,
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape}  dtype={v.dtype}  val={v if v.numel()==1 else v.flatten()[:4]}")
