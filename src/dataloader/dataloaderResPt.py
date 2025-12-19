# feature_loaders.py
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn as nn

from .augmentpt import TensorAugment


class FeatureBagDataset(Dataset):
    """
    Reads .pt files with keys: "feats"(16,2048), "tertile_id"/"tertile_str", etc.
    """
    def __init__(self, dirpath: str,
                 label_from: str = "tertile_id",
                 map_labels: Optional[Dict[str, int]] = None,
                 drop_missing: bool = True,
                 augment=None,                # <--- NEW (callable or None)
                 train: bool = True):         # <--- NEW flag
        self.dir = Path(dirpath)
        self.paths: List[Path] = sorted(self.dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {self.dir}")
        self.label_from = label_from
        self.map_labels = map_labels or {"low": 0, "mid": 1, "high": 2}
        self.drop_missing = drop_missing
        self.augment = augment
        self.train = train

        if drop_missing:
            kept = []
            for p in self.paths:
                obj = torch.load(p, map_location="cpu")
                if self._extract_label(obj) not in (None, -1):
                    kept.append(p)
            if not kept:
                raise RuntimeError(f"All items were filtered out in {self.dir}")
            self.paths = kept

    def _extract_label(self, obj) -> Optional[int]:
        if self.label_from == "tertile_id":
            y = obj.get("tertile_id", None)
            try: return int(y) if y is not None else None
            except: return None
        s = obj.get("tertile_str", None)
        return self.map_labels.get(str(s).lower(), None) if s is not None else None

    def __len__(self): return len(self.paths)

    def __getitem__(self, i: int):
        obj = torch.load(self.paths[i], map_location="cpu")
        x = obj["feats"]                       # (16, 2048)
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        x = x.float()

        if self.train and self.augment is not None:
            x = self.augment(x)                # still (16, 2048)

        y = self._extract_label(obj)
        if y is None: y = -1

        return {
            "feats": x,                        # (16, 2048)
            "label": torch.tensor(int(y)),     # scalar
            "id": obj.get("id", self.paths[i].stem),
            "path": obj.get("path", str(self.paths[i])),
            "tertile_id": obj.get("tertile_id", None),
            "tertile_str": obj.get("tertile_str", None),
        }


class FeatureBagBinDataset(Dataset):
    """
    Reads .pt files with keys: "feats"(16,2048), "tertile_id"/"tertile_str", etc.
    """
    def __init__(self, dirpath: str,
                 label_from: str = "tertile_id",
                 map_labels: Optional[Dict[str, int]] = None,
                 drop_missing: bool = True,
                 augment=None,
                 train: bool = True,
                 include_labels: Optional[set] = None,     # <--- NEW
                 remap: Optional[Dict[int, int]] = None):  # <--- NEW (e.g., {0:0, 2:1})
        self.dir = Path(dirpath)
        self.paths: List[Path] = sorted(self.dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {self.dir}")
        self.label_from = label_from
        self.map_labels = map_labels or {"low": 0, "mid": 1, "high": 2}
        self.drop_missing = drop_missing
        self.augment = augment
        self.train = train
        self.include_labels = set(include_labels) if include_labels else None
        self.remap = remap or {}

        # Build filtered item list: keep (path, y)
        items = []
        for p in self.paths:
            obj = torch.load(p, map_location="cpu")
            y = self._extract_label(obj)
            # if y in (None, -1):
            #     if drop_missing:
            #         continue
            # if self.include_labels is not None and y not in self.include_labels:
            #     continue
            # print("int(y)-------------------------------------", int(y))
            items.append((p, int(y) if y is not None else -1))

        if not items:
            raise RuntimeError(f"No items left after filtering in {self.dir}")
        self.items = items

    def _extract_label(self, obj) -> Optional[int]:
        if self.label_from == "tertile_id":
            # print("obj-----------------------------------", obj.keys())
            y = obj.get("tertile_id")
            # print("_extract_label----------------y--------------------", y)
            return int(y) 

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i: int):
        p, y = self.items[i]
        obj = torch.load(p, map_location="cpu")
        # print("y-------------------------", y )
        x = obj["feats"]
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        x = x.float()

        if self.train and self.augment is not None:
            x = self.augment(x)  # (16, 2048)

        # Remap labels if requested (e.g., {0:0, 2:1})
        if y is None: 
            y = -1
        y = self.remap.get(int(y), int(y))

        # print("in dataloader----------------------------------------")
        return {
            "feats": x,                        
            "label": torch.tensor(int(y)),     
            "id": obj.get("id", Path(p).stem),
            "path": obj.get("path", str(p)),
            "tertile_id": obj.get("tertile_id", None),
            "tertile_str": obj.get("tertile_str", None),
        }



def default_collate(batch):
    """
    Collate dicts into a batch.
    - feats: (B, 16, 2048)
    - label: (B,)
    - others: lists
    """
    feats = torch.stack([b["feats"] for b in batch], dim=0)           # (B,16,2048)
    labels = torch.stack([b["label"] for b in batch], dim=0).long()   # (B,)
    ids = [b["id"] for b in batch]
    paths = [b["path"] for b in batch]
    t_ids = [b["tertile_id"] for b in batch]
    t_strs = [b["tertile_str"] for b in batch]
    return {
        "feats": feats,
        "label": labels,
        "id": ids,
        "path": paths,
        "tertile_id": t_ids,
        "tertile_str": t_strs,
    }

def get_feature_loaders(
    root: str,                              # e.g., "data/BAlt_Expirement/precomputeResNet"
    train_sub: str = "train",
    val_sub: str = "val",
    label_from: str = "tertile_id",         # or "tertile_str"
    map_labels: Optional[Dict[str, int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    shuffle_train: bool = True,
    drop_missing: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders from directories of .pt feature files.
    """
    torch.manual_seed(seed)

    ####################################### augmentation ####################################### 
    aug = TensorAugment(
        p_token_drop=0.2, token_drop_frac=0.15, token_drop_mode="zero",
        p_token_shuffle=0.2, local_shuffle=True, local_window=4,
        p_token_span_mask=0.2, token_span_frac=0.25,
        p_feat_jitter=0.5, feat_jitter_sigma=0.02,
        p_feat_band_mask=0.3, feat_band_frac=0.10,
        p_feat_affine=0.3, affine_scale_std=0.10, affine_shift_std=0.02,
        p_rect_cutout=0.2, rect_token_frac=0.25, rect_feat_frac=0.15
    )
    ## 75.76 %
    aug_light = TensorAugment(  
        p_token_drop=0.05,  token_drop_frac=0.05, token_drop_mode="zero",  # ~drop 1 token occasionally
        p_token_shuffle=0.05, local_shuffle=True, local_window=4,          # rare local shuffle
        p_token_span_mask=0.10, token_span_frac=0.15,                      # short spans
        p_feat_jitter=0.30, feat_jitter_sigma=0.01,                        # milder noise
        p_feat_band_mask=0.15, feat_band_frac=0.05,                        # thinner band masks
        p_feat_affine=0.10, affine_scale_std=0.05, affine_shift_std=0.01,  # small affine jitter
        p_rect_cutout=0.05, rect_token_frac=0.15, rect_feat_frac=0.10      # small rectangles, rare
    )

    aug_light_plus = TensorAugment( ##68.18
        p_token_drop=0.10,  token_drop_frac=0.10,  token_drop_mode="zero",
        p_token_shuffle=0.10, local_shuffle=True,  local_window=4,
        p_token_span_mask=0.15, token_span_frac=0.20,
        p_feat_jitter=0.40,  feat_jitter_sigma=0.015,
        p_feat_band_mask=0.25, feat_band_frac=0.08,
        p_feat_affine=0.15,  affine_scale_std=0.07,  affine_shift_std=0.015,
        p_rect_cutout=0.08,  rect_token_frac=0.20,   rect_feat_frac=0.12,
    )

    aug_strong = TensorAugment( ## bad 
        p_token_drop=0.50,  token_drop_frac=0.30, token_drop_mode="zero",  # drop ~5 tokens on avg
        p_token_shuffle=0.60, local_shuffle=True, local_window=4,          # frequent local reorder
        p_token_span_mask=0.50, token_span_frac=0.40,                      # long masked spans
        p_feat_jitter=0.90, feat_jitter_sigma=0.05,                        # strong Gaussian noise
        p_feat_band_mask=0.60, feat_band_frac=0.20,                        # wide feature bands
        p_feat_affine=0.70, affine_scale_std=0.20, affine_shift_std=0.05,  # larger scaling/shift
        p_rect_cutout=0.50, rect_token_frac=0.40, rect_feat_frac=0.30      # big rectangles, often
    )

    aug_strong_1 = TensorAugment( ## bad 
        p_token_drop=0.40,  token_drop_frac=0.20, token_drop_mode="zero",  # drop ~5 tokens on avg
        p_token_shuffle=0.50, local_shuffle=True, local_window=4,          # frequent local reorder
        p_token_span_mask=0.30, token_span_frac=0.20,                      # long masked spans
        p_feat_jitter=0.80, feat_jitter_sigma=0.05,                        # strong Gaussian noise
        p_feat_band_mask=0.50, feat_band_frac=0.10,                        # wide feature bands
        p_feat_affine=0.50, affine_scale_std=0.10, affine_shift_std=0.05,  # larger scaling/shift
        p_rect_cutout=0.30, rect_token_frac=0.30, rect_feat_frac=0.20      # big rectangles, often
    )


    # Very strong (recommended to try first)
    i = 1.2
    aug_strong = TensorAugment( ## bad 
        p_token_drop=i*0.50,  token_drop_frac=i*0.30, token_drop_mode="zero",  # drop ~5 tokens on avg
        p_token_shuffle=i*0.60, local_shuffle=True, local_window=4,          # frequent local reorder
        p_token_span_mask=i*0.50, token_span_frac=i*0.40,                      # long masked spans
        p_feat_jitter=i*0.90, feat_jitter_sigma=i*0.05,                        # strong Gaussian noise
        p_feat_band_mask=i*0.60, feat_band_frac=i*0.20,                        # wide feature bands
        p_feat_affine=i*0.70, affine_scale_std=i*0.20, affine_shift_std=i*0.05,  # larger scaling/shift
        p_rect_cutout=i*0.50, rect_token_frac=i*0.40, rect_feat_frac=i*0.30      # big rectangles, often
    )
    # Extreme (use with caution)
    aug_extreme = TensorAugment(
        p_token_drop=0.95,  token_drop_frac=0.70, token_drop_mode="zero",
        p_token_shuffle=0.95, local_shuffle=True, local_window=12,
        p_token_span_mask=0.90, token_span_frac=0.75,
        p_feat_jitter=0.98,  feat_jitter_sigma=0.20,
        p_feat_band_mask=0.95, feat_band_frac=0.50,
        p_feat_affine=0.95,  affine_scale_std=0.50, affine_shift_std=0.20,
        p_rect_cutout=0.90, rect_token_frac=0.75, rect_feat_frac=0.60
    )


    aug_light_minus = TensorAugment(
        p_token_drop=0.03,  token_drop_frac=0.05,  token_drop_mode="zero",
        p_token_shuffle=0.03, local_shuffle=True,  local_window=4,
        p_token_span_mask=0.07, token_span_frac=0.12,
        p_feat_jitter=0.25,  feat_jitter_sigma=0.008,
        p_feat_band_mask=0.10, feat_band_frac=0.04,
        p_feat_affine=0.05,  affine_scale_std=0.04,  affine_shift_std=0.008,
        p_rect_cutout=0.03,  rect_token_frac=0.12,   rect_feat_frac=0.08,
    )

    train_ds = FeatureBagBinDataset(
        dirpath=str(Path(root) / train_sub),
        label_from=label_from,
        map_labels=map_labels,
        include_labels={0, 2},     # drop all 1s
        remap={0: 0, 2: 1},        # make it binary: {0,1}        
        drop_missing=drop_missing,
        augment=  aug_strong_1,  #aug_strong, #None, #aug_light, #None, #aug_light, #aug_light_minus, #aug_light_plus, #aug_strong, #aug_light, 
        train=True
    )
 

    
    val_ds = FeatureBagBinDataset(
        dirpath=str(Path(root) / val_sub),
        label_from=label_from,
        include_labels={0, 2},     # drop all 1s
        remap={0: 0, 2: 1},        # make it binary: {0,1}        
        map_labels=map_labels,
        drop_missing=drop_missing,
        augment=None, 
        train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate,
        drop_last=False,
    )
    return train_loader, val_loader
