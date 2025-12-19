# autoencoder_resnet101.py
import os, math, time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    # TorchVision >= 0.13
    from torchvision.models import resnet101, ResNet101_Weights
    _HAS_TV_WEIGHTS = True
except Exception:
    import torchvision
    _HAS_TV_WEIGHTS = False

# ----------------- Utility -----------------
def set_seed(seed: int = 42):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tiny upsample block (deconv + conv) ---
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(self.up(x))

# --- encoder that returns the 2048-D vector using torchvision ResNet101 ---
class ResNet101EncoderVec(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=False):
        super().__init__()
        try:
            from torchvision.models import resnet101, ResNet101_Weights
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            m = resnet101(weights=weights)
        except Exception:
            import torchvision
            m = torchvision.models.resnet101(pretrained=pretrained)

        self.feat_dim = m.fc.in_features  # 2048
        m.fc = nn.Identity()              # keep avgpool; returns 2048 vector
        self.backbone = m
        self.in_channels = in_channels

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        print(f"Using ResNet101, feature dim = {self.feat_dim}")

    def forward(self, x):
        z = self.backbone(x)  # (B, 2048)
        return z

# --- simple decoder: 2048 vector -> image via MLP + deconvs ---
class SimpleVectorDecoder(nn.Module):
    def __init__(self, z_dim=2048, out_ch=3, base_ch=256, base_hw=8, out_activation="sigmoid"):
        super().__init__()
        self.base_ch = base_ch
        self.base_hw = base_hw
        self.out_activation = out_activation

        self.proj = nn.Sequential(
            nn.Linear(z_dim, base_ch * base_hw * base_hw),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpBlock(base_ch, 128)  # 8 -> 16
        self.up2 = UpBlock(128, 64)       # 16 -> 32
        self.up3 = UpBlock(64, 64)        # 32 -> 64
        self.up4 = UpBlock(64, 32)        # 64 -> 128
        self.up5 = UpBlock(32, 32)        # 128 -> 256
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, z, out_size=None):
        b = z.size(0)
        x = self.proj(z).view(b, self.base_ch, self.base_hw, self.base_hw)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.head(x)
        if self.out_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.out_activation == "tanh":
            x = torch.tanh(x)
        return x

# --- full autoencoder wrapper (returns reconstruction; can also return z) ---
class ResNet101VecAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pretrained_encoder=True, out_activation="sigmoid"):
        super().__init__()
        self.encoder = ResNet101EncoderVec(in_channels=in_channels, pretrained=pretrained_encoder)
        self.decoder = SimpleVectorDecoder(z_dim=self.encoder.feat_dim, out_ch=out_channels, out_activation=out_activation)

    def forward(self, x, return_z=False):
        h, w = x.shape[-2:]
        z = self.encoder(x)                # (B, 2048)
        y = self.decoder(z, out_size=(h, w))
        return (y, z) if return_z else y

# ----------------- Factory -----------------
def build_backbone(cfg: dict) -> nn.Module:
    """
    Returns the autoencoder when cfg["backbone"] == "resnet101_ae".
    """
    bb = cfg.get("backbone", "resnet101_ae").lower()
    if bb not in {"resnet101_ae", "resnet101_autoencoder"}:
        raise ValueError(f"Unsupported backbone '{bb}'. Use 'resnet101_ae'.")

    in_ch = cfg.get("in_channels")
    out_ch = cfg.get("out_channels")
    pretrained = cfg.get("pretrained_encoder", True)
    freeze = cfg.get("freeze_encoder", False)
    out_act = cfg.get("out_activation", "sigmoid")  # for [0,1] images use sigmoid

    return ResNet101VecAutoencoder(
        in_channels=in_ch,
        out_channels=out_ch,
        pretrained_encoder=True,
    )


if __name__ == "__main__":
    # ---- Quick smoke test ----
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal config for a 3-channel autoencoder test
    cfg = {
        "backbone": "resnet101_ae",
        "in_channels": 3,
        "out_channels": 1,
        # Set True if you have ImageNet weights cached; otherwise keep False to avoid downloads.
        "pretrained_encoder": False,
        "freeze_encoder": False,
        "out_activation": "sigmoid",  # or None if your inputs aren't [0,1]
    }

    model = build_backbone(cfg).to(device).eval()

    # Print parameter count (millions)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Dummy input
    x = torch.randn(1, 3, 256, 256, device=device)

    # Forward
    y = model(x)

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")

    if y.shape == x.shape:
        print("✓ Shapes match.")
    else:
        print("⚠ Shape mismatch!")
