import torch
import torch.nn as nn

class MLPMixerLayer(nn.Module):
    """
    MLP-Mixer layer for inputs shaped [B, N, D]
      - Token-mixing: mixes across N (sequence/tokens) for each channel D
      - Channel-mixing: mixes across D (features) for each token N
    Output shape: [B, N, D] (same as input)
    """
    def __init__(self, num_tokens: int, dim: int,
                 token_hidden_dim: int = 256, channel_hidden_dim: int = 2048,
                 dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_tokens),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        # Token mixing: apply MLP on N dimension -> [B, D, N] -> [B, D, N] -> back
        y = self.norm1(x)
        y = y.transpose(1, 2)              # [B, D, N]
        y = self.token_mlp(y)              # [B, D, N]
        y = y.transpose(1, 2)              # [B, N, D]
        x = x + y

        # Channel mixing: apply MLP on D dimension
        y = self.norm2(x)
        y = self.channel_mlp(y)            # [B, N, D]
        x = x + y

        return x


class MLPMixer(nn.Module):
    """
    Stacks multiple MLPMixerLayer blocks.
    Input/Output: [B, N, D]
    """
    def __init__(self, num_tokens: int, dim: int,
                 token_hidden_dim: int = 256, channel_hidden_dim: int = 2048,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            MLPMixerLayer(
                num_tokens=num_tokens,
                dim=dim,
                token_hidden_dim=token_hidden_dim,
                channel_hidden_dim=channel_hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------
# Test code (shape check)
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 1
    N = 40000       # number of tokens (choose any fixed N you want)
    D = 1024

    model = MLPMixer(
        num_tokens=N,
        dim=D,
        token_hidden_dim=256,
        channel_hidden_dim=2048,
        num_layers=2,
        dropout=0.0
    )

    x = torch.randn(B, N, D)
    y = model(x)

    print("x shape:", x.shape)  # [1, N, 1024]
    print("y shape:", y.shape)  # [1, N, 1024]
    assert y.shape == x.shape, "Output shape must match input shape!"
    print("✅ Shape preserved.")
