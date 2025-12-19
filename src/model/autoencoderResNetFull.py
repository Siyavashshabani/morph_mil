import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


def freeze_bn_all(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()                           # stop updating running stats
            m.track_running_stats = True       # keep using stored running stats
            if m.affine:
                m.weight.requires_grad_(False) # freeze gamma/beta
                m.bias.requires_grad_(False)

class ResEncoder(nn.Module):
    """
    ResNet-101 encoder with optional freezing of early layers.
    out="map": feature map [B, 2048, H/32, W/32]
    out="vector": 2048-D vector
    out="avgpool": avgpool output [B, 2048, 1, 1]
    """
    def __init__(self, in_channels=3, pretrained=True, freeze_until="layer3", out="map"):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet101(weights=weights)

        # Adapt first conv for custom in_channels
        if in_channels != 3:
            old = m.conv1
            new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new_conv.weight.zero_()
                    new_conv.weight[:, :3].copy_(old.weight)
                    extra = in_channels - 3
                    new_conv.weight[:, 3:].copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, extra, 1, 1))
                else:
                    new_conv.weight[:, :in_channels].copy_(old.weight[:, :in_channels])
            m.conv1 = new_conv

        # Expose ResNet parts
        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool

        self.out = out

        # Freeze everything up to `freeze_until`
        freeze_map = {"stem": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
        target_stage = freeze_map[freeze_until]

        for i, block in enumerate([self.stem, self.layer1, self.layer2, self.layer3, self.layer4]):
            if i <= target_stage:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        h = self.layer4(x)

        if self.out == "map":
            return h
        elif self.out == "avgpool":
            return self.avgpool(h)
        elif self.out == "vector":
            return torch.flatten(self.avgpool(h), 1)
        else:
            raise ValueError(f"Unknown out={self.out}")



# --------- Decoder blocks ---------
class UpBlock(nn.Module):
    """
    Upsample by 2x with ConvTranspose2d, then refine with two 3x3 convs.
    """
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


import torch.nn.functional as F

class ResDecoder(nn.Module):
    """
    Decoder that can start from [B, C, 1, 1] or [B, C, 8, 8].
    If input is 1x1, a ConvTranspose2d seeds it to 8x8, then the usual 2x up blocks run.
    """
    def __init__(self, in_ch=2048, out_ch=3, norm=True, final_activation=None, seed_grid=8):
        super().__init__()
        self.seed_grid = int(seed_grid)

        # 1x1 -> 8x8 (learnable). If input already 8x8, this is skipped in forward.
        self.seed = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=self.seed_grid, stride=self.seed_grid, bias=False),
            nn.BatchNorm2d(in_ch) if norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # regular 2x upsampling stack: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.up1 = UpBlock(in_ch,   1024, norm=norm)
        self.up2 = UpBlock(1024,     512, norm=norm)
        self.up3 = UpBlock(512,      256, norm=norm)
        self.up4 = UpBlock(256,      128, norm=norm)
        self.up5 = UpBlock(128,       64, norm=norm)
        self.head = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

        if final_activation is None:
            self.act = nn.Identity()
        elif final_activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif final_activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("final_activation must be None, 'sigmoid', or 'tanh'.")

    def forward(self, x):
        h, w = x.shape[-2:]
        if (h, w) == (1, 1):
            x = self.seed(x)  # 1x1 -> 8x8
        elif (h, w) != (self.seed_grid, self.seed_grid):
            # safety: if some other size sneaks in, resize to 8x8
            x = F.interpolate(x, size=(self.seed_grid, self.seed_grid), mode="nearest")

        x = self.up1(x)  # 8 -> 16
        x = self.up2(x)  # 16 -> 32
        x = self.up3(x)  # 32 -> 64
        x = self.up4(x)  # 64 -> 128
        x = self.up5(x)  # 128 -> 256
        x = self.head(x)
        return self.act(x)



# --------- Autoencoder wrapper (uses your ResEncoder as-is) ---------
# --------- Autoencoder wrapper (uses your ResEncoder as-is) ---------
class ResFullAutoencoder(nn.Module):
    """
    Uses your existing ResEncoder (returns C5 map) + ResDecoder to reconstruct the image.
    """
    def __init__(self, in_ch=3, out_ch=3, norm=True, final_activation=None, encoder_out="map"):
        super().__init__()
        self.encoder = ResEncoder(in_channels=in_ch, 
                                  pretrained=True, 
                                  freeze_until="layer3", 
                                  out=encoder_out)
        
        self.decoder = ResDecoder(in_ch=2048, 
                                  out_ch=out_ch, 
                                  norm=norm, 
                                  final_activation=final_activation)

        # Freeze BN (stats + affine) in the encoder right away
        self._freeze_bn_all(self.encoder)

    @staticmethod
    def _freeze_bn_all(module: nn.Module):
        """
        Freeze all BatchNorm2d layers: use running stats (no updates) and freeze gamma/beta.
        """
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()                     # stop using batch stats / stop updating running stats
                m.track_running_stats = True # keep existing running stats
                if m.affine:
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def train(self, mode: bool = True):
        """
        Keep normal train/eval for the model, but re-freeze BN in the encoder so
        external calls to .train() don't undo it.
        """
        super().train(mode)
        self._freeze_bn_all(self.encoder)
        return self

    def forward(self, x):
        z = self.encoder(x)   # [B, 2048, 8, 8] for 256x256 inputs (pre-pool C5 map)
        y = self.decoder(z)   # [B, out_ch, 256, 256]
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build the autoencoder
ae = ResFullAutoencoder(in_ch=3,out_ch=3, final_activation="sigmoid").to(device)  # 'sigmoid' if your images are [0,1]

# Test forward
x = torch.randn(2, 3, 256, 256, device=device)
with torch.no_grad():
    y_recon, feats = ae(x)

print("encoder feats:", feats.shape)    # expected: [2, 2048, 8, 8]
print("reconstruction:", y_recon.shape) # expected: [2, 3, 256, 256]
