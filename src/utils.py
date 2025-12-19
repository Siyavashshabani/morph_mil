import math
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def build_attn_maps(
    WSI_attn: torch.Tensor,
    out_hw: Tuple[int, int] = (1024, 1024),
    grid: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    mode: str = "soft",          # "soft" -> bilinear (smooth), "hard" -> blocky/nearest
    eps: float = 1e-6,
):
    """
    Convert WSI_attn [B,P,1] or [B,P] into attention maps and an upsampled heatmap.

    Returns:
        attn_map  : [B, nH, nW]         (row-major)
        attn_norm : [B, nH, nW]         (min-max per sample if normalize=True, else raw)
        attn_ups  : [B, 1, H, W]        (upsampled to out_hw with chosen `mode`)
    Args:
        WSI_attn : attention scores [B,P,1] or [B,P]
        out_hw   : (H,W) output size for upsampled map
        grid     : (nH, nW). If None, infer square grid via sqrt(P).
        normalize: if True, per-sample min-max to [0,1]
        mode     : "soft" (bilinear) or "hard" (blocky/nearest)
        eps      : small constant for numerical stability
    """
    # [B,P,1] -> [B,P]
    attn = WSI_attn.squeeze(-1)
    B, P = attn.shape

    # grid
    if grid is None:
        nH = nW = int(math.sqrt(P))
        assert nH * nW == P, f"P={P} not square; pass grid=(nH,nW)."
    else:
        nH, nW = grid
        assert nH * nW == P, f"grid {grid} does not match P={P}."

    # [B,P] -> [B,nH,nW] (row-major)
    attn_map = attn.view(B, nH, nW)

    # per-sample normalization (inner)
    if normalize:
        mn = attn_map.amin(dim=(1, 2), keepdim=True)
        mx = attn_map.amax(dim=(1, 2), keepdim=True)
        attn_norm = (attn_map - mn) / (mx - mn + eps)
    else:
        attn_norm = attn_map

    # upsample
    H, W = out_hw
    if mode == "soft":
        # smooth heatmap (good for continuous visualization)
        attn_ups = F.interpolate(attn_norm.unsqueeze(1), size=(H, W),
                                 mode="bilinear", align_corners=False)
    elif mode == "hard":
        # exact blocky tiles (no blending). Prefer integer tile replication.
        if (H % nH == 0) and (W % nW == 0):
            ph, pw = H // nH, W // nW
            attn_ups = (attn_norm.unsqueeze(1)
                        .repeat_interleave(ph, dim=2)
                        .repeat_interleave(pw, dim=3))
        else:
            # fallback to nearest if out size not divisible by grid
            attn_ups = F.interpolate(attn_norm.unsqueeze(1), size=(H, W),
                                     mode="nearest")
    else:
        raise ValueError(f"Unknown mode='{mode}'. Use 'soft' or 'hard'.")

    return attn_map, attn_norm, attn_ups

# Example:
# attn_map, attn_norm, attn_ups = build_attn_maps(WSI_attn, out_hw=(1024,1024), mode="hard")  # blocky
# attn_map, attn_norm, attn_ups = build_attn_maps(WSI_attn, out_hw=(1024,1024), mode="soft")  # smooth


class MinMax01:
    def __call__(self, x: torch.Tensor):
        # x: (C,H,W), float in [0,1] or any range
        if x.dtype != torch.float32:
            x = x.float()
        # per-channel min-max
        mn = x.amin(dim=(1,2), keepdim=True)
        mx = x.amax(dim=(1,2), keepdim=True)
        return (x - mn) / (mx - mn + 1e-6)




def load_batch_images(paths, device=None):
    """
    Load images from `paths`, apply `transform`, and stack to [B, 3, H, W].
    - `paths` can be a single str/Path or an iterable of them
    - `transform` should output a tensor of shape [3, H, W] (e.g., Resize->ToTensor->MinMax01)
    - If `device` is given, the batch is moved there
    """
    ## define the transform     
    transform = T.Compose([
        T.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        MinMax01(),
    ])    
    
    # Normalize `paths` to a list
    if isinstance(paths, (str, Path)):
        paths = [paths]

    tensors = []
    for p0 in paths:
        img = Image.open(p0)
        x = transform(img)  # expect [3, 1024, 1024]
        print(f"loaded: {p0} size: {img.size} -> tensor: {tuple(x.shape)}, "
              f"min={x.min():.3f}, max={x.max():.3f}")
        tensors.append(x)

    batch = torch.stack(tensors, dim=0)  # [B, 3, 1024, 1024]
    if device is not None:
        batch = batch.to(device, non_blocking=True)
    return batch



import os, shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm
def save_overlays(imgs_BCHW: torch.Tensor,
                  attn_ups: torch.Tensor,
                  out_dir: str = "overlays",
                  paths=None,
                  labels=None,
                  preds=None,                 # <--- NEW
                  mode: str = "soft",
                  alpha: float = 0.5,
                  cmap: str = "jet"):
    """
    Saves [ RAW | OVERLAY ] tiles per sample.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- shapes ---
    assert imgs_BCHW.ndim == 4 and attn_ups.ndim == 4
    B, C, H, W = imgs_BCHW.shape
    Ba, Ca, Ha, Wa = attn_ups.shape
    assert B == Ba and H == Ha and W == Wa and Ca == 1, "Shape mismatch."

    # --- normalize paths ---
    if isinstance(paths, (str, Path)):
        paths = [paths] * B
    elif paths is None:
        paths = [None] * B
    else:
        assert len(paths) == B, "paths length must match batch size"

    # --- normalize labels ---
    if labels is None:
        labels = [None] * B
    else:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        elif isinstance(labels, np.ndarray):
            labels = labels.tolist()
        labels = list(labels)
        assert len(labels) == B, "labels length must match batch size"

    # --- normalize preds (NEW) ---
    if preds is None:
        preds = [None] * B
    else:
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().tolist()
        elif isinstance(preds, np.ndarray):
            preds = preds.tolist()
        preds = list(preds)
        assert len(preds) == B, "preds length must match batch size"

    cmap_fn = cm.get_cmap(cmap)

    for i in range(B):
        # raw
        img = imgs_BCHW[i].detach().cpu().clamp(0, 1)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        # attention heatmap
        a = attn_ups[i, 0].detach().cpu().clamp(0, 1).numpy()
        heat_rgb = (cmap_fn(a)[..., :3] * 255).round().astype(np.uint8)

        # overlay
        overlay = ((1 - alpha) * img_np.astype(np.float32) + alpha * heat_rgb.astype(np.float32))
        overlay = overlay.round().clip(0, 255).astype(np.uint8)

        # tile [RAW | OVERLAY]
        tile = np.zeros((H, W * 2, 3), dtype=np.uint8)
        tile[:, :W, :] = img_np
        tile[:, W:, :] = overlay

        # filename parts
        stem = (Path(paths[i]).stem if paths[i] is not None else f"sample_{i:03d}")
        y    = labels[i]
        p    = preds[i]

        # cast label/pred cleanly (handles floats/None)
        y_suffix = f"_y{int(y)}" if y is not None else ""
        try:
            p_suffix = f"_p{int(p)}" if p is not None else ""
        except Exception:
            # if p is prob/float array etc., fall back to string-safe
            p_suffix = f"_p{str(p)}" if p is not None else ""

        out_name = f"{stem}{y_suffix}{p_suffix}_{mode}_raw_overlay.png"
        Image.fromarray(tile).save(out_path / out_name)
