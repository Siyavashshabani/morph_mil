import math
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

class DinoV3Backbone(nn.Module):
    """
    Minimal reusable DINOv3 backbone.
      pool:  "cls" | "mean" | "map"
      freeze: if True, params are frozen and eval() is used
    """
    def __init__(
        self,
        repo_dir: str,
        arch: str = "dinov3_vitl16",
        ckpt_path: Optional[str] = None,
        pool: str = "cls",
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        assert pool in {"cls", "mean", "map"}
        self.pool = pool
        self.freeze_backbone = freeze

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 1) load model skeleton from local torch.hub repo
        self.backbone = torch.hub.load(repo_dir, arch, source="local", pretrained=False)
        self.backbone.to(self.device)

        # 2) load checkpoint (optional)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            inc = self.backbone.load_state_dict(state, strict=False)
            if (getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None)):
                print(f"[DinoV3Backbone] loaded with missing={len(getattr(inc,'missing_keys',[]))}, "
                      f"unexpected={len(getattr(inc,'unexpected_keys',[]))}")

        # 3) freeze if requested
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _infer_grid(model, x: torch.Tensor, N: int) -> Tuple[int, int]:
        # Try cached grid_size (some ViT impls set this during forward)
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
            gs = model.patch_embed.grid_size
            if isinstance(gs, (tuple, list)) and len(gs) == 2:
                return int(gs[0]), int(gs[1])

        # Fallback: from patch size and input shape
        ph = pw = None
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
            ps = model.patch_embed.patch_size
            if isinstance(ps, (tuple, list)) and len(ps) == 2:
                ph, pw = int(ps[0]), int(ps[1])
            elif isinstance(ps, int):
                ph = pw = ps
        if ph is not None and pw is not None:
            H, W = x.shape[-2], x.shape[-1]
            return H // ph, W // pw

        # Last fallback: assume square grid from N
        side = int(math.sqrt(N))
        if side * side != N:
            raise ValueError(f"Cannot infer (h,w) from N={N}. Provide square inputs or expose grid_size.")
        return side, side

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          pool="cls"  -> (B, D)
          pool="mean" -> (B, D)
          pool="map"  -> (B, D, h, w)
        """
        x = x.to(self.device, non_blocking=True)
        if self.freeze_backbone:
            self.backbone.eval()

        out = self.backbone.forward_features(x)
        cls_out  = out["x_norm_clstoken"]      # (B, D)
        patches  = out["x_norm_patchtokens"]   # (B, N, D)
        B, N, D  = patches.shape

        if self.pool == "cls":
            return cls_out                      # (B, D)
        elif self.pool == "mean":
            return patches.mean(dim=1)          # (B, D)
        else:  # "map"
            h, w = self._infer_grid(self.backbone, x, N)
            return patches.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)



    # A convenience to know current output dimension without doing a forward
    def out_channels(self, sample_input: Optional[torch.Tensor] = None) -> int:
        if self.pool == "map":
            # channels dimension after optional projection
            if self.proj_map is not None:
                return self.proj_map.out_channels
        else:
            if self.proj_vec is not None:
                return self.proj_vec.out_features

        # If not initialized yet, we need a sample to probe D
        if sample_input is None:
            raise ValueError("Provide sample_input to infer output dimension before first forward.")
        with torch.no_grad():
            out = self.backbone.forward_features(sample_input.to(self.device))
            D = out["x_norm_patchtokens"].shape[-1]
        return self.reduce_to if self.reduce_to is not None else D



# ---- build backbone ----
dino_ckpt = "/home/sshabani/projects/segdino/web_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
repo_dir  = "/home/sshabani/projects/segdino/dinov3"
backbone = DinoV3Backbone(
    repo_dir=repo_dir,
    arch="dinov3_vitl16",
    ckpt_path=dino_ckpt,
    pool="cls",          # "cls" for global CLS; "map" for spatial feature map
    freeze=True           # freeze for fixed feature extraction
)

# ---- use in your training loop ----
B, C, H, W = 8, 3, 256, 256
x = torch.randn(B, C, H, W)               # your batch
with torch.no_grad():                     # remove if you want to finetune (freeze=False)
    z = backbone(x)                       # (B, 256) since pool="mean", reduce_to=256
# print("feat shape:", z.shape)

