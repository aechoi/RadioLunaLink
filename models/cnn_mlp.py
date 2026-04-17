import torch
import torch.nn as nn


class _CoordEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, coord):
        return self.net(coord)


class _CNNBackbone(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 128
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 64
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 32
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                         # 4×4
        )
        self.proj = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class CNNMLPModel(nn.Module):
    """
    Late-fusion CNN + MLP baseline.

    Map -> CNN -> map_feat
    (tx + rx) -> symmetric coord embedding -> coord_feat   [g(tx) + g(rx)]
    [map_feat, coord_feat] -> MLP head -> scalar dBm
    """

    def __init__(self, in_channels=1, embed_dim=256, mlp_hidden=512):
        super().__init__()
        self.backbone = _CNNBackbone(in_channels, embed_dim)
        self.coord_enc = _CoordEncoder(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, hm, tx_coord, rx_coord):
        """
        hm:       (B, C, H, W)
        tx_coord: (B, 2)  normalised to [0, 1]
        rx_coord: (B, 2)  normalised to [0, 1]
        returns:  (B,)    predicted signal strength in dBm
        """
        map_feat = self.backbone(hm)
        # Symmetric: f(A,B) == f(B,A) because addition is commutative
        coord_feat = self.coord_enc(tx_coord) + self.coord_enc(rx_coord)
        return self.head(torch.cat([map_feat, coord_feat], dim=1)).squeeze(1)


if __name__ == "__main__":
    B = 4
    model = CNNMLPModel()
    hm = torch.randn(B, 1, 256, 256)
    tx = torch.rand(B, 2)
    rx = torch.rand(B, 2)
    out = model(hm, tx, rx)
    print(f"output shape: {out.shape}")   # (4,)

    # Verify symmetry
    out_flipped = model(hm, rx, tx)
    max_diff = (out - out_flipped).abs().max().item()
    print(f"symmetry max diff: {max_diff:.2e}")  # should be ~0
