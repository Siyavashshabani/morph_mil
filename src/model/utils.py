import time, math, argparse, random, yaml
import sys, os
# sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from dataloader.dataloader import get_loaders 
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import numpy as np
import torch


# --- Scheduler builder ---
def build_scheduler(optimizer, train_loader, cfg):
    """
    Returns (scheduler, mode) where mode is 'epoch' or 'batch'.
    Supported: cosine, step, onecycle. Use cfg['scheduler'] to select.
    """
    name = str(cfg.get("scheduler", "none")).lower()
    epochs = cfg.get("epochs")
    if name == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs),
            eta_min=float(cfg.get("min_lr", 1e-6)),
        )
        return sch, "epoch"

    if name == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 10)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
        return sch, "epoch"

    if name == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("max_lr", cfg.get("lr", 5e-4) * 10)),
            epochs=int(epochs),
            steps_per_epoch=len(train_loader),
            pct_start=float(cfg.get("warmup_pct", 0.1)),
            anneal_strategy=str(cfg.get("anneal", "cos")).lower(),  # 'cos' or 'linear'
        )
        return sch, "batch"

    if name == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",                                  # lower is better (use val_loss)
            factor=float(cfg.get("factor", 0.85)),       # LR *= factor
            patience=int(cfg.get("patience", 2)),       # epochs with no improvement before reducing
            threshold=float(cfg.get("threshold", 1e-4)),
            threshold_mode=str(cfg.get("threshold_mode", "rel")),  # 'rel' or 'abs'
            cooldown=int(cfg.get("cooldown", 0)),
            min_lr=float(cfg.get("min_lr", 5e-4)),
            # verbose=bool(cfg.get("verbose", True)),
        )
        return sch, "metric"

    return None, None



import torch.nn as nn

def build_loss(cfg: dict) -> nn.Module:
    """
    Returns a PyTorch loss based on cfg['loss'].
    Supported: 'mse', 'l1', 'bce', 'bcewithlogits' (aliases allowed).
    """
    name = str(cfg.get("loss", "mse")).lower()

    if name in ("mse", "l2"):
        return nn.MSELoss()

    if name in ("l1", "mae"):
        return nn.L1Loss()

    if name in ("bce", "binary_cross_entropy"):
        # Use when model outputs are already sigmoid'd to [0,1]
        return nn.BCELoss()

    if name in ("bcewithlogits", "bce_logits", "bcelogits"):
        # Use when model outputs raw logits (no sigmoid in the model)
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unknown loss '{name}'. Use: mse | l1 | bce | bcewithlogits")
