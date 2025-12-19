import os
from collections import defaultdict
import torch
from torchvision.utils import save_image, make_grid

def save_batch_patches(batch, out_dir="patches_out"):
    """
    batch: dict from your DataLoader with keys:
      - image: (N, 1, 256, 256)  in [0,1]
      - path:  list[str] length N (repeated per patch)
      - tertile_str: list[str] length N (repeated per patch)
    Saves:
      - one PNG per patch
      - one 4x4 grid PNG per original image (if it has 16 patches)
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs = batch["image"]           # (N, 1, 256, 256)
    paths = batch["path"]           # list of file paths length N
    labels = batch["tertile_str"]   # list of labels length N

    # group patch indices by original image path
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[p].append(i)

    for orig_path, idxs in groups.items():
        # derive a nice base name
        base = os.path.splitext(os.path.basename(orig_path))[0]
        label = labels[idxs[0]] if labels and idxs else "unknown"

        # save individual patch PNGs
        for j, k in enumerate(idxs):
            # imgs[k] is (1,256,256) and already in [0,1]
            patch_out = os.path.join(out_dir, f"{base}_patch{j:02d}_{label}.png")
            save_image(imgs[k], patch_out)

        # also save a 4x4 grid if exactly 16 patches
        if len(idxs) == 16:
            grid = make_grid(imgs[idxs], nrow=4, padding=2)  # (1, 4*256+pads, 4*256+pads)
            grid_out = os.path.join(out_dir, f"{base}_grid_{label}.png")
            save_image(grid, grid_out)

