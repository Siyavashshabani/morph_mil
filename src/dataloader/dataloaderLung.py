from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# from augment import TensorAugment
from .augment import TensorAugment


def h5_num_patches(h5_path: Path) -> int:
    with h5py.File(str(h5_path), "r") as f:
        if "features" in f:
            return int(f["features"].shape[0])
        if "feats" in f:
            return int(f["feats"].shape[0])
        raise KeyError(f"No 'features' or 'feats' dataset in {h5_path}. Keys: {list(f.keys())}")

def minmax_01_np(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Column-wise MinMax normalization to [0,1].
    Constant columns become all-zeros.
    NaN/inf are converted to 0 before computing min/max.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    denom = np.maximum(mx - mn, eps)

    Xn = (X - mn) / denom

    # If a column is constant (mx==mn), it becomes 0 everywhere (already, due to denom=eps)
    return Xn



def _stem_no_ext(p: Path) -> str:
    return p.name[:-len(p.suffix)] if p.suffix else p.name


def normalize_slide_id_from_pt(pt_path: Path) -> str:
    """
    Example:
      TCGA-3C-AALI-01Z-00-DX1.F6E9....pt  ->  TCGA-3C-AALI-01Z-00-DX1.F6E9....
    We keep the full stem (including UUID part) by default.
    """
    return _stem_no_ext(pt_path)


def normalize_slide_id_from_morph_csv(csv_path: Path) -> str:
    """
    Example:
      TCGA-A8-A0A1-01Z-00-DX1.CA64..._cells_patch_morphometrics.csv
        -> TCGA-A8-A0A1-01Z-00-DX1.CA64...
    """
    stem = _stem_no_ext(csv_path)
    suffix = "_cells_patch_morphometrics"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def normalize_slide_id_generic(x: str) -> str:
    """
    For the labels CSV slide_id column.
    We try to make it comparable to the PT/Morph IDs.

    If your labels CSV uses only the prefix before the dot (no UUID),
    you can switch to returning x.split('.')[0].
    """
    x = str(x).strip()
    # default: keep full id
    return x





import numpy as np
import torch
from typing import Optional

def _h5_to_tensor(x, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    arr = np.asarray(x)

    # Debug (optional)
    # print("shape:", getattr(arr, "shape", None), "dtype:", getattr(arr, "dtype", None), "type:", type(arr))

    # If it's an object array (very common cause of your error), force numeric
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # try to coerce to float32; change to int64 if you expect ints
        arr = arr.astype(np.float32)

    # Ensure plain ndarray + contiguous memory
    if isinstance(arr, np.ndarray):
        arr = np.ascontiguousarray(arr)

        try:
            t = torch.from_numpy(arr)
        except Exception:
            # last resort: convert via Python lists (slower but robust)
            t = torch.tensor(arr.tolist())
    else:
        # scalar / non-array fallback
        t = torch.tensor(arr.item() if hasattr(arr, "item") else arr)

    if dtype is not None:
        t = t.to(dtype)
    return t



def load_h5_embedding(h5_path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    h5_path = Path(h5_path)

    # Safety checks
    if h5_path.suffix not in [".h5", ".hdf5"]:
        raise ValueError(f"Expected .h5/.hdf5 file, got: {h5_path}")
    if not h5py.is_hdf5(str(h5_path)):
        raise ValueError(f"Not a valid HDF5 file (signature not found): {h5_path}")

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if "features" in f:
            feats = _h5_to_tensor(f["features"][...], dtype=torch.float32)
        elif "feats" in f:
            feats = _h5_to_tensor(f["feats"][...], dtype=torch.float32)
        else:
            raise KeyError(f"No 'features' or 'feats' in {h5_path}. Keys: {keys}")

        coords = _h5_to_tensor(f["coords"][...], dtype=torch.int64) if "coords" in f else None
    
    meta: Dict[str, Any] = {
        "h5_keys": keys,
        "features_shape": tuple(feats.shape),
        "coords_shape": tuple(coords.shape) if coords is not None else None,
        "features_dtype": str(feats.dtype),
        "coords_dtype": str(coords.dtype) if coords is not None else None,
    }
    return feats, coords, meta

def load_morph_csv(
    csv_path: Path,
    drop_non_numeric: bool = True,
    keep_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Reads morph CSV. If coord columns exist, returns them too.
    Returns: (X [M,K], coords [M,2] or None, feature_names)
    """
    df = pd.read_csv(csv_path)

    # detect coord columns (common names)
    coord_candidates = [
        ("coord_x", "coord_y"),
        ("x", "y"),
        ("patch_x", "patch_y"),
        ("tile_x", "tile_y"),
    ]
    coords = None
    for cx, cy in coord_candidates:
        if cx in df.columns and cy in df.columns:
            coords = df[[cx, cy]].to_numpy()
            break

    if keep_columns is not None:
        # user-specified columns
        missing = [c for c in keep_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        feat_df = df[keep_columns].copy()
    else:
        feat_df = df.copy()

    if drop_non_numeric:
        # remove obvious non-feature cols
        # (we keep coords separately if present)
        non_feature_cols = set()
        if coords is not None:
            non_feature_cols |= {cx, cy}
        # also drop common ID-like columns if present
        for c in ["slide_id", "wsi", "filename", "patch_id", "tile_id", "cell_id"]:
            if c in feat_df.columns:
                non_feature_cols.add(c)

        feat_df = feat_df.drop(columns=[c for c in non_feature_cols if c in feat_df.columns], errors="ignore")
        feat_df = feat_df.select_dtypes(include=[np.number])

    feature_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=np.float32)

    return X, coords, feature_names


def align_morph_to_coords(
    bag_coords: torch.Tensor,  # [N,2]
    morph_X: np.ndarray,       # [M,K]
    morph_coords: np.ndarray,  # [M,2]
    fill_value: float = 0.0,
) -> np.ndarray:
    if bag_coords.ndim != 2 or bag_coords.shape[1] != 2:
        raise ValueError("bag_coords must be [N,2]")

    # Ensure numeric, remove NaNs in morph features
    morph_X = np.asarray(morph_X, dtype=np.float32)
    morph_X = np.nan_to_num(morph_X, nan=0.0, posinf=0.0, neginf=0.0)

    # Build mapping: (x,y) -> mean morph vector (handles duplicates)
    dfm = pd.DataFrame(morph_X)
    dfm["x"] = morph_coords[:, 0].astype(np.int64)
    dfm["y"] = morph_coords[:, 1].astype(np.int64)
    grouped = dfm.groupby(["x", "y"], sort=False).mean(numeric_only=True)

    bc = bag_coords.detach().cpu().numpy()
    bcx = bc[:, 0].astype(np.int64)
    bcy = bc[:, 1].astype(np.int64)

    out = np.full((bc.shape[0], morph_X.shape[1]), fill_value, dtype=np.float32)

    for i in range(bc.shape[0]):
        key = (bcx[i], bcy[i])
        if key in grouped.index:
            out[i] = grouped.loc[key].to_numpy(dtype=np.float32)

    return out




from collections import defaultdict

@dataclass
class BRCAItem:
    slide_id: str
    h5_path: Path
    label: int


class BRCAEmbedDataset(Dataset):
    def __init__(
        self,
        h5_dir: str | Path,
        labels_csv: str | Path,
        label_col: str = "label",
        slide_id_col: str = "slide_id",
        strict: bool = True,
        max_patches: int = 40000,
        aug_flag: bool = True,
        # keep_morph_columns / align_by_coords_if_possible removed
    ):
        self.h5_dir = Path(h5_dir)
        self.labels_csv = Path(labels_csv)

        self._rng = None  # set per-worker (see worker_init_fn below)
        self._base_seed = 42
        self.aug_replace = True
        
        if not self.h5_dir.exists():
            raise FileNotFoundError(f"h5_dir not found: {self.h5_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")

        self.strict = strict

        # ---- load labels ----
        df_lab = pd.read_csv(self.labels_csv)
        if slide_id_col not in df_lab.columns or label_col not in df_lab.columns:
            raise ValueError(
                f"labels_csv must contain columns {slide_id_col!r} and {label_col!r}. "
                f"Got: {list(df_lab.columns)}"
            )

        labels_map: Dict[str, Any] = {}
        for _, r in df_lab.iterrows():
            sid = normalize_slide_id_generic(r[slide_id_col])
            # print("sid-------------------", sid)
            labels_map[sid] = r[label_col]

        # ---- validate labels ----
        allowed = {"luad", "lusc"}
        uniq = set(labels_map.values())
        bad = uniq - allowed
        if bad:
            raise ValueError(f"Unexpected labels found: {bad}. Expected only {allowed}.")

        self.label_to_int = {"luad": 0, "lusc": 1}
        self.int_to_label = {0: "luad", 1: "lusc"}

        # ---- index H5 files ----
        h5_map: Dict[str, Path] = {}

        for p in sorted(list(self.h5_dir.glob("*.h5")) + list(self.h5_dir.glob("*.hdf5"))):
            key = p.name.split(".", 1)[0]   # or p.stem.split(".", 1)[0]
            h5_map[key] = p

        # ---- build items: (h5, label) only ----
        keys = set(h5_map.keys())

        items: List[BRCAItem] = []
        missing_label = 0

        for sid in sorted(keys):
            label_val = None
            if sid in labels_map:
                label_val = labels_map[sid]
            else:
                sid_prefix = sid.split(".")[0]
                if sid_prefix in labels_map:
                    label_val = labels_map[sid_prefix]

            if label_val is None:
                missing_label += 1
                if strict:
                    continue
                label_int = -1
            else:
                label_int = self.label_to_int[label_val]

            items.append(BRCAItem(slide_id=sid, h5_path=h5_map[sid], label=label_int))

        print("len(self.items)----------------------------", len(items))
        if strict and len(items) == 0:
            raise RuntimeError(
                "No matched items found. Likely slide_id formatting mismatch.\n"
                f"Example H5 key: {next(iter(h5_map.keys())) if h5_map else 'NONE'}\n"
                f"Example Label key: {next(iter(labels_map.keys())) if labels_map else 'NONE'}\n"
                "Tip: If your labels slide_id column has no UUID part, it may only match sid.split('.')[0]."
            )

        self.items = items

        # ---- drop slides with too many patches ----
        if max_patches is not None:
            kept: List[BRCAItem] = []
            dropped: List[tuple[str, int]] = []

            for it in self.items:
                n = h5_num_patches(it.h5_path)  # cheap: reads only dataset shape
                if n <= max_patches:
                    kept.append(it)
                else:
                    dropped.append((it.slide_id, n))

            if dropped:
                print(f"[Dataset] Dropped {len(dropped)} slides with > {max_patches} patches.")
                for sid, n in dropped[:10]:
                    print(f"  - {sid}: {n}")

            self.items = kept


        # ✅ ADD THIS RIGHT HERE (after filtering, so indices match final items)
        self.label_to_indices = defaultdict(list)
        for k, it in enumerate(self.items):
            if it.label >= 0:              # skip unknown if strict=False
                self.label_to_indices[it.label].append(k)


        self._missing_label = missing_label


        ## 
        self.label_to_indices = defaultdict(list)
        for k, it in enumerate(self.items):
            # optional: skip unknown labels
            if it.label >= 0:
                self.label_to_indices[it.label].append(k)


        # ---- augmentation ----
        self.aug_flag = aug_flag
        self.aug = TensorAugment(
            p_token_drop=0.45, token_drop_frac=0.30, token_drop_mode="zero",
            p_token_shuffle=0.40, local_shuffle=True, local_window=8,
            p_token_span_mask=0.45, token_span_frac=0.45,
            p_feat_jitter=0.80, feat_jitter_sigma=0.06,
            p_feat_band_mask=0.55, feat_band_frac=0.20,
            p_feat_affine=0.55, affine_scale_std=0.20, affine_shift_std=0.05,
            p_rect_cutout=0.45, rect_token_frac=0.45, rect_feat_frac=0.30
        )

        self.aug_light = TensorAugment(
            p_token_drop=0.20, token_drop_frac=0.12, token_drop_mode="zero",
            p_token_shuffle=0.15, local_shuffle=True, local_window=6,
            p_token_span_mask=0.25, token_span_frac=0.25,
            p_feat_jitter=0.55, feat_jitter_sigma=0.03,
            p_feat_band_mask=0.30, feat_band_frac=0.10,
            p_feat_affine=0.25, affine_scale_std=0.10, affine_shift_std=0.02,
            p_rect_cutout=0.20, rect_token_frac=0.25, rect_feat_frac=0.18
        )


    def mix_replace_tokens_one_view(
        self,
        feats: torch.Tensor,
        feats_donor: torch.Tensor,
        *,
        replace_frac: float = 0.25,
    ) -> torch.Tensor:
        self._ensure_rng()

        if feats.dim() != 2 or feats_donor.dim() != 2:
            raise ValueError(f"Expected [N,D]. Got {tuple(feats.shape)} and {tuple(feats_donor.shape)}")
        if feats.size(1) != feats_donor.size(1):
            raise ValueError(f"D mismatch: {feats.size(1)} vs {feats_donor.size(1)}")

        Na = int(feats.shape[0])
        Nd = int(feats_donor.shape[0])
        if Na == 0 or Nd == 0:
            return feats.clone()

        k = max(1, int(round(float(replace_frac) * Na)))
        k = min(k, Na)

        idx_a = self._rng.choice(Na, size=k, replace=False)
        idx_d = self._rng.choice(Nd, size=k, replace=True)

        out = feats.clone()
        idx_a_t = torch.as_tensor(idx_a, device=out.device, dtype=torch.long)
        idx_d_t = torch.as_tensor(idx_d, device=out.device, dtype=torch.long)
        out[idx_a_t] = feats_donor[idx_d_t]
        return out

    def _ensure_rng(self):
        if self._rng is None:
            # single-process fallback (num_workers=0)
            self._rng = np.random.default_rng(self._base_seed)
            
    def _sample_same_label_index(self, label: int, avoid: set[int]) -> int:
        """
        Sample an index from the same-label pool, avoiding any indices in `avoid`.
        Falls back to a non-avoided index if pool is small.
        """
        pool = self.label_to_indices[label]
        if len(pool) == 0:
            return next(iter(avoid))  # should never happen if labels_to_indices built correctly

        # if there is at least one candidate not in avoid
        candidates = [k for k in pool if k not in avoid]
        if len(candidates) > 0:
            return int(self._rng.choice(candidates))

        # otherwise (pool too small), just return something from pool
        return int(self._rng.choice(pool))


    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_rng()
        it = self.items[idx]

        feats, coords, h5_meta = load_h5_embedding(it.h5_path)  # [Na, D]

        if self.aug_flag and self.aug_replace:
            # pick two different same-label donors
            j = self._sample_same_label_index(it.label, avoid={idx})
            k = self._sample_same_label_index(it.label, avoid={idx, j})

            it_b = self.items[j]
            it_c = self.items[k]

            feats_b, _, _ = load_h5_embedding(it_b.h5_path)
            feats_c, _, _ = load_h5_embedding(it_c.h5_path)

            feats_aug1 = self.mix_replace_tokens_one_view(feats, feats_b, replace_frac=0.5)
            feats_aug2 = self.mix_replace_tokens_one_view(feats, feats_c, replace_frac=0.5)

            feats = torch.stack([feats, feats_aug1, feats_aug2], dim=0)  # [3, Na, D]

        elif self.aug_flag:
            # your original augmentation path (if you still want it)
            feats_aug1 = self.aug(feats.clone())
            feats_aug2 = self.aug_light(feats.clone())
            feats = torch.stack([feats, feats_aug1, feats_aug2], dim=0)

        return {
            "slide_id": it.slide_id,
            "feats": feats,
            "label": it.label,
            "coords": coords,
            "h5_meta": h5_meta,
            "h5_path": str(it.h5_path),
        }
import torch
from typing import List, Dict, Any, Tuple
import numpy as np
from torch.utils.data import Subset, DataLoader


# -------------------------------------------------------------------
# Collate (morph removed)
# -------------------------------------------------------------------
def clam_like_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    MIL collate:
    - batch_size==1: return tensors directly
    - batch_size>1: return lists for variable-length bags.
    """
    assert len(batch) >= 1

    if len(batch) == 1:
        b = batch[0]
        return {
            "slide_id": b["slide_id"],
            "feats": b["feats"],  # Tensor [N,D] or [3,N,D] if aug enabled
            "label": torch.tensor([b["label"]], dtype=torch.long),  # [1]

            "coords": b.get("coords", None),     # Tensor [N,2] or None
            "h5_path": b.get("h5_path", None),
            "h5_meta": b.get("h5_meta", None),
        }

    # batch_size > 1 (variable-length bags)
    return {
        "slide_id": [b["slide_id"] for b in batch],
        "feats": [b["feats"] for b in batch],  # list of [Ni,D] or [3,Ni,D]
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),  # [B]

        "coords": [b.get("coords", None) for b in batch],
        "h5_path": [b.get("h5_path", None) for b in batch],
        "h5_meta": [b.get("h5_meta", None) for b in batch],
    }


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _get_sid_from_dataset(ds, i: int) -> str:
    # If ds is a Subset, map i -> original index
    if isinstance(ds, Subset):
        base_ds = ds.dataset
        base_i = ds.indices[i]
        return _get_sid_from_dataset(base_ds, base_i)

    # Your dataset has .items with .slide_id
    if hasattr(ds, "items"):
        return str(ds.items[i].slide_id)

    # Fallback: use __getitem__ dict
    sample = ds[i]
    if isinstance(sample, dict) and "slide_id" in sample:
        return str(sample["slide_id"])

    raise RuntimeError("Can't infer slide id from dataset.")




# -------------------------------------------------------------------
# Train/val/test loaders (morph removed)
# -------------------------------------------------------------------
import numpy as np
from torch.utils.data import Subset, DataLoader
from typing import Any, Dict, Tuple

def train_val_loaders(
    h5_dir: str,
    labels_csv: str,
    *,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,   # NEW: random test split too
    seed: int = 42,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    strict: bool = True,
    aug_flag: bool = True,
    max_patches: int = 40000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns: train_loader, val_loader, test_loader
    Random splits across all items (no filename-based test set).
    """

    ds = BRCAEmbedDataset(
        h5_dir=h5_dir,
        labels_csv=labels_csv,
        slide_id_col="slide",
        label_col="label",
        strict=strict,
        max_patches=max_patches,
        aug_flag=aug_flag,
    )

    n = len(ds)
    if n < 3:
        raise ValueError(f"Dataset too small for train/val/test split: n={n}")

    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0):
        raise ValueError("Require: val_ratio>=0, test_ratio>=0, and val_ratio + test_ratio < 1.0")

    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))

    # make sure each split is non-empty when possible
    n_test = max(1, n_test) if n >= 3 and test_ratio > 0 else n_test
    n_val  = max(1, n_val)  if n >= 3 and val_ratio  > 0 else n_val
    if n_test + n_val >= n:
        # fallback: guarantee at least 1 train
        n_test = min(n_test, n - 2)
        n_val  = min(n_val,  n - 1 - n_test)

    test_idx = all_idx[:n_test].tolist()
    val_idx  = all_idx[n_test:n_test + n_val].tolist()
    train_idx = all_idx[n_test + n_val:].tolist()

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    test_ds  = Subset(ds, test_idx)

    print("Example train sids:", [_get_sid_from_dataset(ds, i) for i in train_idx[:5]])
    print("Example val sids:  ", [_get_sid_from_dataset(ds, i) for i in val_idx[:5]])
    print("Example test sids: ", [_get_sid_from_dataset(ds, i) for i in test_idx[:5]])
    print(f"Counts: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clam_like_collate,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clam_like_collate,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=clam_like_collate,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    h5_dir = "/home/sshabani/projects/snuffy/datasets/tcga/encoded_uni/h5_files"
    labels_csv = "/home/sshabani/projects/snuffy/datasets/tcga/single/patients.csv"

    train_loader, val_loader, test_loader = train_val_loaders(
        h5_dir=h5_dir,
        labels_csv=labels_csv,
        val_ratio=0.2,
        seed=42,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        strict=True,
        aug_flag=True,
        max_patches= 50000,
    )

    batch = next(iter(train_loader))
    print("train batch slide:", batch["slide_id"], "label:", batch["label"][0].item())