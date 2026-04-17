import math
import torch
import torch.nn as nn


def _sinusoidal_2d(h, w, dim):
    """
    Returns (h*w, dim) grid of 2D sinusoidal positional encodings.
    First dim//2 channels encode row, last dim//2 encode col.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin encoding"
    half = dim // 2
    d = half // 2
    div = torch.exp(torch.arange(d) * (-math.log(10000) / (d - 1)))  # (d,)

    rows = torch.arange(h).float().unsqueeze(1) * div  # (h, d)
    cols = torch.arange(w).float().unsqueeze(1) * div  # (w, d)

    row_enc = torch.cat([rows.sin(), rows.cos()], dim=1)  # (h, half)
    col_enc = torch.cat([cols.sin(), cols.cos()], dim=1)  # (w, half)

    # Broadcast to (h, w, dim)
    grid = torch.cat([
        row_enc.unsqueeze(1).expand(h, w, half),
        col_enc.unsqueeze(0).expand(h, w, half),
    ], dim=2)
    return grid.reshape(h * w, dim)  # (h*w, dim)


def _sinusoidal_coord(xy, dim, max_val=1.0):
    """
    Encode normalised (x, y) coords in [0, 1] to (B, dim) sinusoidal embeddings.
    xy: (B, 2)
    """
    assert dim % 4 == 0
    d = dim // 4
    div = torch.exp(torch.arange(d, device=xy.device) * (-math.log(10000) / max(d - 1, 1)))
    x = xy[:, 0:1] * div   # (B, d)
    y = xy[:, 1:2] * div   # (B, d)
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)  # (B, dim)


class _PatchEmbed(nn.Module):
    """Non-overlapping patch projection + 2D sinusoidal PE, outputs (B, N, dim)."""

    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_h = img_size // patch_size
        self.n_w = img_size // patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        pe = _sinusoidal_2d(self.n_h, self.n_w, embed_dim)  # (N, dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, dim)
        tokens = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        return tokens + self.pe


class _CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from coords, keys/values from patch tokens."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # q: (B, Q, dim)   kv: (B, N, dim)
        return self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         need_weights=False)[0]


class CoordCondViT(nn.Module):
    """
    Coordinate-conditioned ViT with cross-attention.

    1. Patch-embed the heightmap → N patch tokens with 2D sinusoidal PE.
    2. Self-attention encoder builds long-range terrain context.
    3. TX and RX coords are sinusoidally encoded, summed (symmetric), and
       used as the query in a cross-attention layer over patch tokens.
    4. MLP head → scalar dBm.

    forward(hm, tx_coord, rx_coord) matches the CNNMLPModel interface.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=1,
        embed_dim=256,
        n_heads=8,
        n_layers=6,
        mlp_hidden=512,
        dropout=0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0
        assert embed_dim % 4 == 0

        self.patch_embed = _PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm, more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projects sinusoidal coord embedding to embed_dim
        self.coord_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )

        self.cross_attn = _CrossAttention(embed_dim, n_heads)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.GELU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def _encode_coord(self, xy):
        """(B, 2) → (B, 1, embed_dim) query token."""
        return self.coord_proj(_sinusoidal_coord(xy, self.coord_proj[0].in_features))

    def forward(self, hm, tx_coord, rx_coord):
        """
        hm:       (B, 1, H, W)
        tx_coord: (B, 2)  normalised to [0, 1]
        rx_coord: (B, 2)  normalised to [0, 1]
        returns:  (B,)    predicted signal strength in dBm
        """
        # Patch tokens with spatial PE
        tokens = self.patch_embed(hm)              # (B, N, dim)
        tokens = self.encoder(tokens)               # (B, N, dim)

        # Symmetric coord query: g(TX) + g(RX)  →  f(A,B) == f(B,A)
        tx_q = self._encode_coord(tx_coord)         # (B, dim)
        rx_q = self._encode_coord(rx_coord)         # (B, dim)
        query = (tx_q + rx_q).unsqueeze(1)          # (B, 1, dim)

        # Cross-attend query over terrain tokens
        attended = self.cross_attn(query, tokens)   # (B, 1, dim)

        return self.head(attended.squeeze(1)).squeeze(1)  # (B,)


if __name__ == "__main__":
    B = 4
    model = CoordCondViT()
    hm = torch.randn(B, 1, 256, 256)
    tx = torch.rand(B, 2)
    rx = torch.rand(B, 2)
    out = model(hm, tx, rx)
    print(f"output shape: {out.shape}")

    out_flipped = model(hm, rx, tx)
    print(f"symmetry max diff: {(out - out_flipped).abs().max().item():.2e}")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n:,}")
