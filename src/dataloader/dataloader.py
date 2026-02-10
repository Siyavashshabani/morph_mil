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
            feats = torch.from_numpy(f["features"][:])
        elif "feats" in f:
            feats = torch.from_numpy(f["feats"][:])
        else:
            raise KeyError(f"No 'features' or 'feats' in {h5_path}. Keys: {keys}")

        coords = torch.from_numpy(f["coords"][:]) if "coords" in f else None

    meta = {
        "h5_keys": keys,
        "features_shape": tuple(feats.shape),
        "coords_shape": tuple(coords.shape) if coords is not None else None,
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

@dataclass
class BRCAItem:
    slide_id: str
    h5_path: Path
    morph_path: Path
    label: int

class BRCAEmbedMorphDataset(Dataset):
    def __init__(
        self,
        h5_dir: str | Path,
        morph_dir: str | Path,
        labels_csv: str | Path,
        label_col: str = "label",
        slide_id_col: str = "slide_id",
        keep_morph_columns: Optional[List[str]] = None,
        align_by_coords_if_possible: bool = True,
        strict: bool = True,
        max_patches: int = 60000,   # <-- add this
        aug_flag: bool = True,
    ):
        self.h5_dir = Path(h5_dir)
        self.morph_dir = Path(morph_dir)
        self.labels_csv = Path(labels_csv)

        if not self.h5_dir.exists():
            raise FileNotFoundError(f"h5_dir not found: {self.h5_dir}")
        if not self.morph_dir.exists():
            raise FileNotFoundError(f"morph_dir not found: {self.morph_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")

        self.keep_morph_columns = keep_morph_columns
        self.align_by_coords_if_possible = align_by_coords_if_possible
        self.strict = strict

        # ---- load labels ----
        df_lab = pd.read_csv(self.labels_csv)
        if slide_id_col not in df_lab.columns or label_col not in df_lab.columns:
            raise ValueError(
                f"labels_csv must contain columns {slide_id_col!r} and {label_col!r}. "
                f"Got: {list(df_lab.columns)}"
            )

        # map slide_id -> label (raw first)
        labels_map: Dict[str, Any] = {}
        for _, r in df_lab.iterrows():
            sid = normalize_slide_id_generic(r[slide_id_col])
            labels_map[sid] = r[label_col]

        # ---- FIXED binary mapping: IDC / ILC ----
        # Normalize labels to avoid "idc", " IDC ", etc.
        labels_map = {k: str(v).strip().upper() for k, v in labels_map.items()}

        allowed = {"IDC", "ILC"}
        uniq = set(labels_map.values())
        bad = uniq - allowed
        if bad:
            raise ValueError(f"Unexpected labels found: {bad}. Expected only {allowed}.")

        # Choose your encoding (IDC=0, ILC=1)
        self.label_to_int = {"IDC": 0, "ILC": 1}
        self.int_to_label = {0: "IDC", 1: "ILC"}

        # ---- index H5 files ----
        h5_map: Dict[str, Path] = {}
        for p in sorted(self.h5_dir.glob("*.h5")):
            h5_map[p.stem] = p
        for p in sorted(self.h5_dir.glob("*.hdf5")):
            h5_map[p.stem] = p
            
        # ---- index morph CSV files ----
        morph_map: Dict[str, Path] = {}
        for c in sorted(self.morph_dir.glob("*.csv")):
            sid = normalize_slide_id_from_morph_csv(c)
            morph_map[sid] = c

        # ---- build items: intersection of (pt, morph, label) ----
        keys = set(h5_map.keys()) & set(morph_map.keys())

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
                    continue  # skip
                else:
                    label_int = -1
            else:
                # label_val is now guaranteed "IDC" or "ILC"
                label_int = self.label_to_int[label_val]

            items.append(
                BRCAItem(
                    slide_id=sid,
                    h5_path=h5_map[sid],
                    morph_path=morph_map[sid],
                    label=label_int,
                )
            )
        print("len(self.items)----------------------------", len(items))
        if strict and len(items) == 0:
            raise RuntimeError(
                "No matched items found. Likely slide_id formatting mismatch.\n"
                f"Example PT key: {next(iter(pt_map.keys())) if pt_map else 'NONE'}\n"
                f"Example Morph key: {next(iter(morph_map.keys())) if morph_map else 'NONE'}\n"
                f"Example Label key: {next(iter(labels_map.keys())) if labels_map else 'NONE'}\n"
                "Tip: If your labels slide_id column has no UUID part, it may only match sid.split('.')[0]."
            )

        self.items = items

        # ---- drop slides with too many patches ----
        if max_patches is not None:
            kept: List[BRCAItem] = []
            dropped: List[tuple[str, int]] = []

            for it in items:
                n = h5_num_patches(it.h5_path)  # cheap: reads only dataset shape
                if n <= max_patches:
                    kept.append(it)
                else:
                    dropped.append((it.slide_id, n))

            if len(dropped) > 0:
                print(f"[Dataset] Dropped {len(dropped)} slides with > {max_patches} patches.")
                # show a few
                for sid, n in dropped[:10]:
                    print(f"  - {sid}: {n}")

            items = kept        
        
        
        self._missing_label = missing_label

        
        ## augmentation 
        self.aug_flag = aug_flag 
        self.aug = TensorAugment(
            # drop more tokens, more often
            p_token_drop=0.45, token_drop_frac=0.30, token_drop_mode="zero",

            # shuffle more often; increase local disruption
            p_token_shuffle=0.40, local_shuffle=True, local_window=8,

            # mask longer spans more often
            p_token_span_mask=0.45, token_span_frac=0.45,

            # stronger feature noise
            p_feat_jitter=0.80, feat_jitter_sigma=0.06,

            # mask more feature bands
            p_feat_band_mask=0.55, feat_band_frac=0.20,

            # stronger scale/shift jitter
            p_feat_affine=0.55, affine_scale_std=0.20, affine_shift_std=0.05,

            # cut out bigger rectangles more often
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
        

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        feats, coords, h5_meta = load_h5_embedding(it.h5_path)  # feats: [N,D], coords: [N,2] or None
        morph_X, morph_coords, morph_names = load_morph_csv(
            it.morph_path,
            drop_non_numeric=True,
            keep_columns=self.keep_morph_columns,
        )

        # Normalize morph features to [0,1] BEFORE alignment
        morph_X = minmax_01_np(morph_X)

        # Align morph to ALL patches in H5 (missing patches -> zeros)
        if self.align_by_coords_if_possible and (coords is not None) and (morph_coords is not None):
            morph_aligned = align_morph_to_coords(
                bag_coords=coords,        # ✅ coords from h5
                morph_X=morph_X,
                morph_coords=morph_coords,
                fill_value=0.0,
            )
            morph_tensor = torch.from_numpy(morph_aligned)  # [N,K]
            morph_is_aligned = True
        else:
            morph_tensor = torch.from_numpy(morph_X)        # [M,K]
            morph_is_aligned = False



        ### adding the augmetations
        if self.aug_flag: 
            feats_aug1 = self.aug(feats.clone())
            feats_aug2 = self.aug_light(feats.clone())
            feats = torch.stack([feats, feats_aug1, feats_aug2], dim=0)  # [3, N, D]
            # it.label = torch.as_tensor(it.label, dtype=torch.long).repeat(3)

        return {
            "slide_id": it.slide_id,

            # what your training loop uses
            "feats": feats,
            "label": it.label,

            # extra info (kept, not necessarily used in training loop)
            "coords": coords,
            "morph": morph_tensor,
            "morph_feature_names": morph_names,
            "morph_aligned": morph_is_aligned,

            "h5_meta": h5_meta,
            "h5_path": str(it.h5_path),
            "morph_path": str(it.morph_path),
        }


import torch
from typing import List, Dict, Any

def clam_like_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    MIL collate:
    - batch_size==1: return tensors directly so training loop can do:
        x = batch["feats"].to(device)
        y = batch["label"].to(device)
    - batch_size>1: return lists for variable-length bags.
    """
    assert len(batch) >= 1

    if len(batch) == 1:
        b = batch[0]
        return {
            "slide_id": b["slide_id"],
            "feats": b["feats"],                         # Tensor [N,D]
            "label": torch.tensor([b["label"]], dtype=torch.long),  # shape [1]

            "coords": b.get("coords", None),             # Tensor [N,2] or None
            "morph": b.get("morph", None),               # Tensor [N,K] or [M,K]
            "morph_feature_names": b.get("morph_feature_names", []),
            "morph_aligned": b.get("morph_aligned", False),

            "h5_path": b.get("h5_path", None),
            "morph_path": b.get("morph_path", None),
            "h5_meta": b.get("h5_meta", None),
        }

    # batch_size > 1 (variable-length bags)
    return {
        "slide_id": [b["slide_id"] for b in batch],
        "feats": [b["feats"] for b in batch],  # list of [Ni,D]
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),  # [B]

        "coords": [b.get("coords", None) for b in batch],
        "morph": [b.get("morph", None) for b in batch],
        "morph_feature_names": batch[0].get("morph_feature_names", []) if len(batch) else [],
        "morph_aligned": [b.get("morph_aligned", False) for b in batch],

        "h5_path": [b.get("h5_path", None) for b in batch],
        "morph_path": [b.get("morph_path", None) for b in batch],
        "h5_meta": [b.get("h5_meta", None) for b in batch],
    }



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################



def train_val_loaders(
    h5_dir: str,
    morph_dir: str,
    labels_csv: str,
    *,
    val_ratio: float = 0.3,
    seed: int = 42,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    keep_morph_columns=None,
    align_by_coords_if_possible: bool = True,
    strict: bool = True,
    use_weighted_sampler: bool = True, 
    aug_flag: bool = True,
    max_patches: int = 50000
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Returns: train_loader, val_loader, info_dict
    Splits at slide level (items) using random permutation.
    """

    ds = BRCAEmbedMorphDataset(
        h5_dir=h5_dir,
        morph_dir=morph_dir,
        labels_csv=labels_csv,
        slide_id_col="slide_id",
        label_col="label",
        keep_morph_columns=keep_morph_columns,
        align_by_coords_if_possible=align_by_coords_if_possible,
        strict=strict,
        aug_flag= aug_flag,
        max_patches = max_patches
    )
    
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty after matching. Check ID formats / paths.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * val_ratio)))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if len(train_idx) == 0:
        # if dataset is tiny, ensure at least 1 train sample
        train_idx = val_idx[:-1]
        val_idx = val_idx[-1:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     shuffle=shuffle_train,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     collate_fn=clam_like_collate,
    #     drop_last=False,
    # )
    # ---- class-weighted sampling for TRAIN only ----
    if use_weighted_sampler:
        # train_ds is a Subset, train_ds.indices are indices into ds.items
        train_labels = np.array([ds.items[i].label for i in train_ds.indices], dtype=np.int64)

        class_counts = np.bincount(train_labels, minlength=2)  # [count0, count1]
        # inverse-frequency weights (common choice)
        class_weights = 1.0 / np.maximum(class_counts, 1)

        sample_weights = class_weights[train_labels]  # one weight per training sample

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),   # samples per epoch
            replacement=True,                  # oversample minority
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,       # ✅ use sampler instead of shuffle
            shuffle=False,         # must be False when sampler is set
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=clam_like_collate,
            drop_last=False,
        )
    else:
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

    # helpful stats
    def _count_labels(subset: Subset) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for i in subset.indices:
            y = ds.items[i].label
            counts[y] = counts.get(y, 0) + 1
        return counts

    info = {
        "n_total": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "label_to_int": getattr(ds, "label_to_int", None),
        "int_to_label": getattr(ds, "int_to_label", None),
        "train_label_counts": _count_labels(train_ds),
        "val_label_counts": _count_labels(val_ds),
    }

    return train_loader, val_loader, info


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    h5_dir = "/home/sshabani/projects/CLAM/data_BRCA/data_BRCA_regression/encoded_uni_no_normalization/h5_files"
    morph_dir = "/home/sshabani/projects/CLAM/data_BRCA/test_resolution/encoded_morphs_csv"
    labels_csv = "/home/sshabani/projects/CLAM/data_BRCA/preprocess/slides_with_labels_IDC_ILC_with_mag.csv"

    train_loader, val_loader, info = train_val_loaders(
        h5_dir=h5_dir,
        morph_dir=morph_dir,
        labels_csv=labels_csv,
        val_ratio=0.2,
        seed=42,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    print("Split info:", info)
    batch = next(iter(train_loader))
    print("train batch slide:", batch["slide_id"], "label:", batch["label"][0].item())
