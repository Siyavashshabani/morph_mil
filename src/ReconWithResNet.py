#!/usr/bin/env python3
"""
Training script with a Trainer class.
"""

import time, math, argparse, random, yaml
import sys, os
# sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataloader import get_loaders 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os, time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import timm
from torchvision.utils import make_grid, save_image

from model.autoencoderRes import ResNet101VecAutoencoder
from model.autoencoder import Autoencoder
from model.autoencoderResNetFull import ResFullAutoencoder
from model.utils import build_scheduler, build_loss
# ---------- Utilities ----------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def accuracy_logits(logits, targets):
    if logits.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return (preds == targets).sum().item() / targets.numel()

def save_ckpt(state, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

# ---------- Model ----------
import torch.nn as nn
from torchvision.models import resnet50, resnet18,resnet101 

# ----------------- Factory -----------------
def build_backbone(cfg: dict) -> nn.Module:
    """
    Returns the autoencoder when cfg["backbone"] == "resnet101_ae".
    """
    if cfg.get("backbone") == "resnet101_ae":
        bb = cfg.get("backbone", "resnet101_ae").lower()
        if bb not in {"resnet101_ae", "resnet101_autoencoder"}:
            raise ValueError(f"Unsupported backbone '{bb}'. Use 'resnet101_ae'.")

        in_ch = cfg.get("in_channels")
        out_ch = cfg.get("out_channels")
        pretrained = cfg.get("pretrained_encoder", True)
        freeze = cfg.get("freeze_encoder", False)
        out_act = cfg.get("out_activation", "sigmoid")  # for [0,1] images use sigmoid
        return ResNet101VecAutoencoder(
            in_channels=in_ch,
            out_channels=out_ch,
            pretrained_encoder=True,
        )
    elif cfg.get("backbone") == "autoencoder":
        in_ch = cfg.get("in_channels")
        out_ch = cfg.get("out_channels")
        out_act = cfg.get("out_activation", "sigmoid") 
        return Autoencoder(
            in_channels=in_ch,
            out_channels=out_ch,
            enc_channels=(64, 128, 256, 512, 1024),   # 5 downs after the stem
            latent_dim=2048,
            base_grid=7,
            decoder_widths=(512, 256, 128, 64, 32),
            out_activation=out_act,            
            )
    elif cfg.get("backbone") == "resautoencoder":
        in_ch = cfg.get("in_channels")
        out_ch = cfg.get("out_channels")
        out_act = cfg.get("out_activation", "sigmoid")
        enc_out =  cfg.get("encoder_out")
        return ResFullAutoencoder(in_ch=in_ch,
                                  out_ch=out_ch, 
                                  final_activation=out_act, 
                                  encoder_out=enc_out
                                  )

# ----------------- Trainer -----------------
class Trainer:
    def __init__(self, cfg):
        """
        get_loaders should be a callable returning (train_loader, val_loader).
        Each batch can be X or (X, y); for AE we reconstruct X.
        """
        self.cfg = cfg
        set_seed(cfg.get("seed", 42))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader, self.val_loader = get_loaders(
            folder=cfg["folder"],
            excel_path=cfg["excel"],
            id_column=cfg.get("id_column", "SampleID"),
            tertile_column=cfg.get("tertile_column", "Tertile"),
            pattern=cfg.get("pattern", "**/*DAPI.tif"),
            val_ratio=cfg.get("val_ratio", 0.2),
            batch_size=cfg.get("batch_size", 4),
            num_workers=cfg.get("num_workers", 2),
            seed=cfg.get("seed", 42),
        )

        # Model / Optim / Loss
        self.model = build_backbone(cfg).to(self.device)
        
        ## loss function 
        self.loss_fn = build_loss(cfg)

        ## scheduler
        lr = cfg.get("lr", 5e-4)
        wd = cfg.get("wd", 1e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler, self._sched_mode = build_scheduler(self.optimizer, self.train_loader, cfg)

        ## Checkpoints
        self.outdir = cfg.get("outdir", "checkpoints")
        os.makedirs(self.outdir, exist_ok=True)
        self.ckpt_best = os.path.join(self.outdir, "best.pt")
        self.ckpt_last = os.path.join(self.outdir, "last.pt")

        self.best_val_loss = float("inf")

        # AMP toggle
        self.use_amp = bool(cfg.get("amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        print(self.model.__class__.__name__,
              "params:", sum(p.numel() for p in self.model.parameters())/1e6, "M")

        # TensorBoard
        logdir   = self.cfg.get("tb_logdir")
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        print(f"[TensorBoard] writing events to: {os.path.abspath(logdir)}")


    @torch.no_grad()
    def test(self, use_best: bool = True, batches: int = 1, nrow: int = 8):
        # 1) load weights
        ckpt_path = self.ckpt_best if use_best else self.ckpt_last
        print("load the model from this ckpt_path---------------:", ckpt_path)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"], strict=True)
            print(f"[Test] loaded {ckpt_path} (epoch={ckpt.get('epoch','?')})")
        else:
            print(f"[Test] checkpoint not found: {ckpt_path} — using current weights.")

        self.model.eval()
        outdir = os.path.join(self.outdir, "test_samples")
        os.makedirs(outdir, exist_ok=True)

        # use val_loader as test source (simple)
        done = 0
        for i, batch in enumerate(self.train_loader, 1):
            x = batch["image"].float().to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_hat = self.model(x)
                x_hat = y_hat[0] if isinstance(y_hat, (tuple, list)) else y_hat

            # clamp to [0,1] for visualization and ensure 3ch
            xin  = x.detach().cpu().clamp(0, 1)
            xrec = x_hat.detach().cpu().clamp(0, 1)
            to3 = (lambda t: t.repeat(1,3,1,1) if t.size(1)==1 else t[:, :3])

            k = min(xin.size(0), nrow * 2)
            grid_in  = make_grid(to3(xin[:k]),  nrow=min(nrow, k))
            grid_rec = make_grid(to3(xrec[:k]), nrow=min(nrow, k))

            save_image(grid_in,  os.path.join(outdir, f"batch{i:03d}_input.png"))
            save_image(grid_rec, os.path.join(outdir, f"batch{i:03d}_recon.png"))
            print(f"[Test] saved batch {i} grids → {outdir}")

            # optional: log to TensorBoard
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_image(f"test/input_b{i:03d}", grid_in)
                self.writer.add_image(f"test/recon_b{i:03d}", grid_rec)
                self.writer.flush()

            done += 1
            if done >= batches:
                break

    def _unpack_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        return x

    # --- Trainer methods ---
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total, count = 0.0, 0

        for batch in self.train_loader:
            x = batch["image"].float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_hat = self.model(x)
                x_hat = y_hat[0] if isinstance(y_hat, (tuple, list)) else y_hat
                loss = self.loss_fn(x_hat, x)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Per-batch schedulers (e.g., OneCycleLR) — step **without** a metric
            if self.scheduler and getattr(self, "_sched_mode", None) == "batch":
                self.scheduler.step()

            bs = x.size(0)
            total += loss.item() * bs
            count += bs

        avg = total / max(count, 1)
        print(f"[Epoch {epoch}] train_loss: {avg:.4f}")
        self.writer.add_scalar("loss/train", avg, epoch)

        # epoch LR (after any batch-stepping)
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("lr", lr, epoch)
        print(f"[Epoch {epoch}] learning rate: {lr:.6f}")
        return avg



    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total, count = 0.0, 0

        for batch in self.val_loader:
            x = batch["image"].float().to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_hat = self.model(x)
                x_hat = y_hat[0] if isinstance(y_hat, (tuple, list)) else y_hat
                loss = self.loss_fn(x_hat, x)
            bs = x.size(0)
            total += loss.item() * bs
            count += bs

        avg = total / max(count, 1)
        print(f"[Epoch {epoch}] val_loss: {avg:.4f}")
        self.writer.add_scalar("loss/val", avg, epoch)

        # Step ReduceLROnPlateau here with the **averaged** val loss
        if self.scheduler and getattr(self, "_sched_mode", None) == "metric":
            self.scheduler.step(avg )
            # (optional) log LR after LR may have changed
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", lr, epoch)

        return avg

    def fit(self):
        epochs = int(self.cfg.get("epochs", 20))
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss   = self.validate(epoch)

            # ---- Scheduler step (after val) ----
            if self.scheduler:
                mode = getattr(self, "_sched_mode", None)  # 'epoch' | 'metric' | 'batch'
                if mode == "epoch":          # e.g., CosineAnnealingLR, StepLR
                    self.scheduler.step()
                elif mode == "metric":       # e.g., ReduceLROnPlateau (use val loss)
                    self.scheduler.step(val_loss)
                # ('batch' schedulers like OneCycleLR are stepped inside train_one_epoch)

            # optional: log current LR at epoch end
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr/epoch_end", lr, epoch)
            self.writer.flush()

            # ---- Checkpoints ----
            torch.save({"model": self.model.state_dict(), "cfg": self.cfg, "epoch": epoch}, self.ckpt_last)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg, "epoch": epoch}, self.ckpt_best)
                print(f"  ✓ New best (val_loss={val_loss:.4f}) saved to {self.ckpt_best}")

        self.writer.close()



# ----------------- main code -----------------
if __name__ == "__main__":

    # Load config
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./src/ReconWithResNet.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    ## define the trainer 
    trainer = Trainer(cfg)
        
    ## train 
    if cfg["mode"] == "train":    
        trainer.fit()

    ## test and plot the output 
    elif cfg["mode"] == "test":
        trainer.test(use_best=cfg.get("use_best", True),
                    batches=cfg.get("test_batches", 1),
                    nrow=cfg.get("test_nrow", 8))
    else:
        raise ValueError(f"Unknown mode: {mode}")    
