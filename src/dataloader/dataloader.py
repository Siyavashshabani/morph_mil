# pip install pandas openpyxl tifffile torch torchvision

import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import tifffile
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import InterpolationMode  # <-- add this
from pathlib import Path
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Subset
import random, re

_PNUM_RE = re.compile(r"[._-]p(\d+)", flags=re.IGNORECASE)  # matches ...p123...


import torch
from torchvision import transforms as T
try:
    from torchvision.transforms import InterpolationMode as IM
except Exception:
    from PIL import Image as IM  # use IM.BILINEAR, IM.NEAREST, etc.

from .utils import save_batch_patches
# --- per-image min-max normalization to [0,1]
class MinMax01(object):
    def __call__(self, x):  # x: (C,H,W) float tensor
        xmin = x.amin(dim=(1,2), keepdim=True)
        xmax = x.amax(dim=(1,2), keepdim=True)
        return (x - xmin) / (xmax - xmin).clamp(min=1e-6)

# --- patchify into 256×256 tiles ---
class Patchify256(object):
    """
    Split CHW tensor into 256x256 patches.
    If grid is None, infer grid from H,W (must be multiples of 256).
    1024x1024 -> 4x4 -> 16 patches.
    Returns (P, C, 256, 256).
    """
    def __init__(self, grid=None, interpolation=IM.BILINEAR):
        self.grid = grid
        self.interp = interpolation

    def __call__(self, x):  # x: (C,H,W)
        C, H, W = x.shape

        # If a specific grid is requested, resize to its total size
        if self.grid is not None:
            gh, gw = self.grid
            target_h, target_w = gh * 256, gw * 256
            if (H, W) != (target_h, target_w):
                x = T.Resize((target_h, target_w), interpolation=self.interp)(x)
                C, H, W = x.shape

        # Must be divisible by 256
        if H % 256 != 0 or W % 256 != 0:
            raise ValueError(f"Image size {(H,W)} not divisible by 256. Resize/pad first.")

        nH, nW = H // 256, W // 256
        patches = (
            x.unfold(1, 256, 256)            # (C, nH, 256, W)
             .unfold(3, 256, 256)            # (C, nH, 256, nW, 256)
             .permute(1, 3, 0, 2, 4)         # (nH, nW, C, 256, 256)
             .contiguous()
             .view(nH * nW, C, 256, 256)     # (P, C, 256, 256)
        )
        return patches



class OverlappedPatchify256(object):
    """
    Split a CHW tensor into 256x256 patches with overlap.
    Returns (P, C, 256, 256). For 1024x1024 and overlap=0.5 -> 7x7=49 patches.

    Args:
        overlap (float): fraction of patch that overlaps with neighbors, in [0,1).
                         0.5 -> stride 128 for 256x256 patches.
        grid (tuple[int,int] | None): if given (gh,gw), first resize to (gh*256, gw*256).
        interpolation: interpolation mode used for optional resize.
        pad_mode (str): 'reflect' | 'replicate' | 'constant' for edge padding if needed.
        pad_value (float): used only if pad_mode='constant'.
    """
    def __init__(self, overlap=0.5, grid=None, interpolation=IM.BILINEAR,
                 pad_mode="reflect", pad_value=0.0):
        assert 0.0 <= overlap < 1.0, "overlap must be in [0,1)"
        self.patch = 256
        self.stride = max(1, int(round(self.patch * (1.0 - overlap))))
        self.grid = grid
        self.interp = interpolation
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def __call__(self, x):  # x: (C,H,W), torch.Tensor
        C, H, W = x.shape

        # Optional: resize to exact grid size
        if self.grid is not None:
            gh, gw = self.grid
            target_h, target_w = gh * self.patch, gw * self.patch
            if (H, W) != (target_h, target_w):
                x = T.Resize((target_h, target_w), interpolation=self.interp)(x)
                C, H, W = x.shape

        # Compute required padding so that unfold covers the full extent with the chosen stride
        K = self.patch
        S = self.stride

        def needed_pad(L):
            if L < K:
                return K - L
            rem = (L - K) % S
            return 0 if rem == 0 else (S - rem)

        pad_h = needed_pad(H)
        pad_w = needed_pad(W)

        if pad_h or pad_w:
            # F.pad uses (left, right, top, bottom)
            if self.pad_mode == "constant":
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=self.pad_value)
            else:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)
            C, H, W = x.shape  # update after padding

        # Unfold with overlap (stride S)
        patches = (
            x.unfold(1, K, S)                 # (C, nH, K, W)
             .unfold(3, K, S)                 # (C, nH, K, nW, K)
             .permute(1, 3, 0, 2, 4)          # (nH, nW, C, K, K)
             .contiguous()
             .view(-1, C, K, K)               # (P, C, 256, 256)
        )

        return patches



# --- collate: flatten patches into batch dim ---
def collate_patches_as_batch(batch):
    imgs = torch.cat([b["image"] for b in batch], dim=0)   # (sum_P, C, 256, 256)
    tertile_ids = torch.cat([
        torch.full((b["image"].shape[0],), b["tertile_id"], dtype=torch.long)
        for b in batch
    ])
    tertile_strs = sum([[b["tertile_str"]]*b["image"].shape[0] for b in batch], [])
    paths = sum([[b["path"]]*b["image"].shape[0] for b in batch], [])
    return {"image": imgs, "tertile_id": tertile_ids, "tertile_str": tertile_strs, "path": paths}




import torch
from PIL import Image

class ReplicateTo3Channels:
    def __call__(self, x):
        # PIL image path
        if isinstance(x, Image.Image):
            # ensure it's single-channel first (L) then convert to RGB
            if x.mode not in ("L", "I;16", "I"):
                # already multi-channel; leave it as-is
                return x
            return x.convert("RGB")

        # Torch tensor path
        if isinstance(x, torch.Tensor):
            # accept [H,W] → [1,H,W]
            if x.ndim == 2:
                x = x.unsqueeze(0)
            # replicate if single-channel
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.repeat(3, 1, 1)  # explicit copy; safer than expand
            return x

        # Fallback: return input unchanged
        return x


def extract_pnum_from_filename(path: str):
    """Return the numeric p-id (as string) from a filename or None if not found."""
    m = _PNUM_RE.search(os.path.basename(path))
    # print("m-----------------------",m, type(m))
    return m.group(1) if m else None

def extract_pnum_from_sampleid(s: str):
    """Return the numeric p-id (as string) from a SampleID string like 'X1b.p203'."""
    if s is None:
        return None
    s = str(s)
    m = _PNUM_RE.search(s)
    return m.group(1) if m else None

def normalize_tertile(val: str):
    """Map various strings to {'low','mid','high'}."""
    if val is None:
        return None
    s = str(val).strip().lower()
    # normalize spaces/underscores and Greek beta → 'b'
    s = s.replace("β", "b").replace(" ", "").replace("_", "")
    # allow variants like 'lowbalt', 'high-balt', etc.
    if "low" in s:
        return "low"
    if "mid" in s or "medium" in s:
        return "mid"
    if "high" in s:
        return "high"
    return None

class TiffDataset(Dataset):
    def __init__(
        self,
        folder,
        excel_path,
        pattern="**/*DAPI*.tif",   # recursive
        id_column="SampleID",      # column with strings like 'X1b.p203'
        tertile_column="Tertile",  # column with Low/Mid/High (or variants)
        transform=None,
        keep_labels=("low", "high"),   # <--- NEW: only keep these
        binarize=True,                 # <--- NEW: map low->0, high->1
    ):
        # 1) Collect image files
        pat = re.compile(r"DAPI(?:_\d+)?\.(?:tif|tiff)$", flags=re.IGNORECASE)
        all_tifs = list(Path(folder).rglob("*.tif")) + list(Path(folder).rglob("*.tiff"))
        self.files = sorted({str(p) for p in all_tifs if pat.search(p.name)})
        if not self.files:
            raise FileNotFoundError(f"No TIFFs found under {folder}")
        self.transform = transform

        # 2) Load Excel and build pnum -> tertile mapping
        df = pd.read_excel(excel_path)
        df["_pnum"] = df[id_column].map(extract_pnum_from_sampleid)
        df["_tertile_norm"] = df[tertile_column].map(normalize_tertile)

        # Build mapping; if duplicates exist, keep the first non-null label
        p2tertile = {}
        for _, row in df.iterrows():
            p = row["_pnum"]
            t = row["_tertile_norm"]
            if p and t and p not in p2tertile:
                p2tertile[p] = t
        self.p2tertile = p2tertile

        # 3) Filter files to ONLY the labels we want (e.g., low/high)
        keep = set(keep_labels) if keep_labels else None
        if keep:
            filtered = []
            for f in self.files:
                pnum = str(int(extract_pnum_from_filename(f)))
                t = self.p2tertile.get(pnum, None)
                if t in keep:
                    filtered.append(f)
            self.files = filtered
            if not self.files:
                raise RuntimeError("After filtering, no files remain (check keep_labels and Excel mapping).")

        # 4) Label mapping (binary or 3-class)
        if binarize:
            # only low/high expected due to filtering
            self.label_to_id = {"low": 0, "high": 1}
        else:
            self.label_to_id = {"low": 0, "mid": 1, "high": 2}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = tifffile.imread(path)  # assume 2D grayscale
        if img.ndim == 2:
            img = img[None, ...]     # (1,H,W)
        img = torch.from_numpy(img.astype(np.float32))

        if self.transform:
            img = self.transform(img)

        # match by p-number from filename
        pnum = str(int(extract_pnum_from_filename(path)))
        tertile_str = self.p2tertile.get(pnum, None)

        # guard: in case something slipped through
        if tertile_str not in self.label_to_id:
            # you can either raise, or mark as missing; raising is safer
            raise KeyError(f"Label '{tertile_str}' for pnum={pnum} not in mapping {list(self.label_to_id.keys())}")

        tertile_id = self.label_to_id[tertile_str]

        return {
            "image": img,
            "path": path,
            "pnum": pnum,                   # e.g., '757'
            "tertile_id": tertile_id,       # 0/1 if binarize=True
            "tertile_str": tertile_str,     # 'low' or 'high'
        }

# --- per-image min-max normalization to [0,1]
class MinMax01(object):
    def __call__(self, x):  # x: (C,H,W)
        xmin = x.amin(dim=(1,2), keepdim=True)
        xmax = x.amax(dim=(1,2), keepdim=True)
        return (x - xmin) / (xmax - xmin).clamp(min=1e-6)

from torch.utils.data import random_split, DataLoader



def get_loaders(
    folder,
    excel_path,
    id_column="SampleID",
    tertile_column="Tertile",
    pattern="**/*DAPI*.tif",
    val_ratio=0.2,
    batch_size=1,
    num_workers=1,
    seed=42,
):
    """
    Build train/val dataloaders for TIFF dataset with patchification.
    """
    transform = T.Compose([
        T.Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR),
        MinMax01(),
        ReplicateTo3Channels(),    # make [1,H,W] → [3,H,W]
        Patchify256(grid=None),   # 1024 -> 16 patches of 256x256
        # OverlappedPatchify256(overlap=0.5),
    ])

    # Full dataset
    ds = TiffDataset(
        folder=folder,
        excel_path=excel_path,
        pattern=pattern,
        id_column=id_column,
        tertile_column=tertile_column,
        transform=transform
    )

    ################################################# Split
    # n_total = len(ds)
    # n_val = int(val_ratio * n_total)
    # n_train = n_total - n_val
    # train_ds, val_ds = random_split(
    #     ds,
    #     [n_train, n_val],
    #     generator=torch.Generator().manual_seed(seed)
    # )
# --- grouped split: keep all samples from the same tumor ID in the same split ---
# --- grouped split by tumor ID token like "p275R" (case-insensitive) ---

    ################################################# Split
    def _tumor_id_from_path(p: str) -> str:
        """
        Extracts tumor ID token like p275R / P0203 from filename.
        Matches: p + digits + optional trailing letters (e.g., R).
        Examples:
        A1819-p275R-03_DAPI.tif           -> p275R
        A1819-P0203-4MGLTumor-1_DAPI.tif  -> P0203
        """
        stem = Path(p).stem
        m = re.search(r'(?i)\b(p\d+[a-z]*)\b', stem)   # case-insensitive
        return m.group(1) if m else stem               # fallback if pattern missing

    # Build groups (may touch ds[i], but only uses 'path')
    group_to_idxs = defaultdict(list)
    for i in range(len(ds)):
        sample = ds[i]
        gid = _tumor_id_from_path(sample["path"])
        group_to_idxs[gid].append(i)
        sample = None  # drop ref to big tensors, if any

    groups = list(group_to_idxs.keys())

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(groups)

    # Target sizes
    n_total = len(ds)
    n_val   = int(round(val_ratio * n_total))

    # Greedy assign whole groups to approach target val size
    val_groups, train_groups = [], []
    val_count = 0
    for g in groups:
        gsz = len(group_to_idxs[g])
        if abs((val_count + gsz) - n_val) <= abs(val_count - n_val):
            val_groups.append(g); val_count += gsz
        else:
            train_groups.append(g)

    # Indices per split
    val_indices   = [i for g in val_groups   for i in group_to_idxs[g]]
    train_indices = [i for g in train_groups for i in group_to_idxs[g]]

    train_ds = Subset(ds, train_indices)
    val_ds   = Subset(ds, val_indices)

    print(f"[split] train={len(train_ds)}  val={len(val_ds)}  "
        f"groups: train={len(train_groups)} val={len(val_groups)}")



    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_patches_as_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_patches_as_batch,
    )

    return train_loader, val_loader




if __name__ == "__main__":
    train_loader, val_loader = get_loaders(
        folder="/home/sshabani/projects/balt_experiment/data/BAlt_Expirement",
        excel_path="/home/sshabani/projects/balt_experiment/data/BAlt_Expirement/bAlt_scores_complete.xlsx",
        val_ratio=0.2,
        batch_size=1,
        num_workers=1
    )

    # Example: one batch from train
    batch = next(iter(train_loader))
    print("Train batch:", batch["image"].shape, batch["tertile_id"].shape, ) # batch["path"][:2]

    ## save the patches 
    save_batch_patches(batch, out_dir="/home/sshabani/projects/balt_experiment/output/test_loader")  # saves PNGs here
    
    # Example: one batch from val
    batch_val = next(iter(val_loader))
    print("Val batch:", batch_val["image"].shape, batch_val["tertile_id"].shape, ) #batch_val["path"][:2]
