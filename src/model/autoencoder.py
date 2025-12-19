# conv_vec_autoencoder.py
from typing import Iterable, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Strided conv downsample + conv refine."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ds = ConvBNAct(in_ch, out_ch, k=3, s=2, p=1)   # /2
        self.cv = ConvBNAct(out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x):
        return self.cv(self.ds(x))

class UpBlock(nn.Module):
    """Nearest upsample + conv refine."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.cv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.cv(x)

class Autoencoder(nn.Module):
    """
    Fully-convolutional vector-bottleneck autoencoder (latent_dim=2048).
    No third-party backbone.

    Args:
        in_channels:  input channels (e.g., 3)
        out_channels: output channels (usually same as input)
        enc_channels: encoder stage widths (before final 1x1 to latent_dim)
        latent_dim:   size of latent vector (default=2048)
        base_grid:    seed H=W for decoder feature map
        decoder_widths: channel widths for decoder from coarse->fine
        out_activation: 'sigmoid', 'tanh', or None
        freeze_encoder: if True, encoder params won't train
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_channels: Iterable[int] = (64, 128, 256, 512, 1024),
        latent_dim: int = 2048,
        base_grid: int = 7,
        decoder_widths: Iterable[int] = (512, 256, 128, 64, 32),
        out_activation: str | None = "sigmoid",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.base_grid  = int(base_grid)

        # ----- Encoder -----
        chs = list(enc_channels)
        stem = [ConvBNAct(in_channels, chs[0], k=7, s=2, p=3)]  # initial /2
        downs = []
        for i in range(len(chs) - 1):
            downs.append(DownBlock(chs[i], chs[i+1]))
        self.enc_stem = nn.Sequential(*stem)
        self.enc_body = nn.ModuleList(downs)
        self.enc_head = nn.Conv2d(chs[-1], self.latent_dim, kernel_size=1, bias=False)
        self.enc_gap  = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 2048, 1, 1)

        if freeze_encoder:
            for p in self.enc_stem.parameters(): p.requires_grad = False
            for p in self.enc_body.parameters(): p.requires_grad = False
            for p in self.enc_head.parameters(): p.requires_grad = False

        # ----- Projection to decoder seed -----
        self.dec_seed_channels = decoder_widths[0]
        self.proj = nn.Linear(self.latent_dim, self.dec_seed_channels * self.base_grid * self.base_grid)

        # ----- Decoder -----
        ups = []
        widths = list(decoder_widths)
        for i in range(len(widths) - 1):
            ups.append(UpBlock(widths[i], widths[i + 1]))
        self.up_blocks = nn.ModuleList(ups)
        self.dec_head  = nn.Conv2d(widths[-1], out_channels, kernel_size=3, padding=1)

        if out_activation is None:
            self.out_act = nn.Identity()
        elif out_activation.lower() == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_activation.lower() == "tanh":
            self.out_act = nn.Tanh()
        else:
            raise ValueError("out_activation must be one of {'sigmoid','tanh',None}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    # -------- API --------
    # @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-D latent vector."""
        return self._encode_to_vec(x)

    def decode(self, z: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Decode latent vectors z -> images of size target_hw=(H,W)."""
        H, W = target_hw
        B = z.shape[0]
        y = self.proj(z).view(B, self.dec_seed_channels, self.base_grid, self.base_grid)
        for block in self.up_blocks:
            y = block(y)
        y = self.dec_head(y)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return self.out_act(y)

    def forward(self, x: torch.Tensor):
        """Return (reconstruction, latent)."""
        H, W = x.shape[-2], x.shape[-1]
        z = self._encode_to_vec(x)           # (B, 2048)
        x_hat = self.decode(z, (H, W))       # (B, out_channels, H, W)
        return x_hat, z

    # -------- internals --------
    def _encode_to_vec(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc_stem(x)                 # /2
        for blk in self.enc_body:            # further downsamples
            y = blk(y)
        y = self.enc_head(y)                 # -> (B, 2048, h, w)
        y = self.enc_gap(y)                  # -> (B, 2048, 1, 1)
        return torch.flatten(y, 1)           # -> (B, 2048)

def _load_state(path: str, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)  # support raw state_dict too
    # handle DDP "module." prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    epoch = ckpt.get("epoch", "?")
    return state, epoch

import torch, torch.nn as nn

model = Autoencoder(
    in_channels=3,
    out_channels=3,
    enc_channels=(64, 128, 256, 512, 1024),   # 5 downs after the stem
    latent_dim=2048,
    base_grid=7,
    decoder_widths=(512, 256, 128, 64, 32),
    out_activation="sigmoid",
)

x = torch.randn(8, 3, 256, 256)
x_hat, z = model(x)
print(z.shape)      # torch.Size([8, 2048])
print(x_hat.shape)  # torch.Size([8, 3, 256, 256])

loss = nn.L1Loss()(x_hat, x)
loss.backward()
