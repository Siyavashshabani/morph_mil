#!/usr/bin/env python3
"""
Training script with a Trainer class.
"""
import torch.nn.functional as F

import time, math, argparse, random, yaml
import sys, os
# sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataloader import train_val_loaders 
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np, os

import numpy as np
import numpy as np
import torch
import os, yaml, math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.distributed import DistributedSampler
from model.morphMIL import MorphMIL
from model.poolMIL import PoolMIL
from model.utils import build_scheduler, build_loss

######################################################################################
######################################################################################
def ddp_setup():
    """
    Returns: (is_ddp, rank, world_size, local_rank)
    Works with: torchrun --nproc_per_node=...
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
   
   
def save_ckpt(self, path, epoch, val_acc):
    if not self._rank0:
        return
    model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
    torch.save({
        "cfg": self.cfg,
        "epoch": epoch,
        "model": model_state,
        "optimizer": self.optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)

def load_ckpt(self, path):
    ckpt = torch.load(path, map_location=self.device)
    target = self.model.module if isinstance(self.model, DDP) else self.model
    target.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        self.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
######################################################################################
######################################################################################
     
        
        
def _to_numpy(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def _maybe_softmax(probs_like):
    # If rows don't sum to ~1, treat as logits and softmax them
    import torch
    p = probs_like
    if isinstance(p, np.ndarray):
        pt = torch.from_numpy(p)
    else:
        pt = p
    if pt.ndim == 2:
        rowsum = pt.float().softmax(dim=1).sum(dim=1)  # we’ll softmax anyway if needed
        # Just return softmaxed always; safe even if they're already probs.
        pt = pt.float().softmax(dim=1)
    elif pt.ndim == 1:
        # For binary case we can keep as scores; AUC can use raw scores.
        return _to_numpy(pt)
    return _to_numpy(pt)



def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))

def _softmax_2c(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def compute_binary_auc_and_ap(y_true, y_prob_or_logits):
    """
    Accepts:
      y_true: (N,) with labels {0,1}
      y_prob_or_logits: (N,), (N,1), or (N,2); logits or probabilities; torch/np/list OK
    Returns: (roc_auc or None, pr_auc or None, y_score)   # y_score is pos-class prob in [0,1]
    """
    # print("y_true, y_prob_or_logits---------------", y_true.shape, y_prob_or_logits.shape)
    # ---- normalize inputs
    try:
        import torch
        if isinstance(y_prob_or_logits, torch.Tensor):
            y_prob_or_logits = y_prob_or_logits.detach().cpu().numpy()
    except Exception:
        pass

    y_true = np.asarray(y_true).astype(int)

    y = y_prob_or_logits
    if isinstance(y, (list, tuple)):
        # concatenate per-batch outputs
        y = np.concatenate([np.asarray(a).squeeze() for a in y], axis=0)
    y = np.asarray(y)

    # ---- unify to a 1-D positive-class score
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]  # single-logit or single-prob
        elif y.shape[1] == 2:
            looks_like_probs = (
                np.all(y >= 0) and np.all(y <= 1) and np.allclose(y.sum(axis=1), 1, atol=1e-3)
            )
            y = y[:, 1] if looks_like_probs else _softmax_2c(y)[:, 1]
        else:
            # try squeeze if there are singleton dims; else error
            y = np.squeeze(y)
            if y.ndim > 1:
                raise ValueError(f"Expected (N,), (N,1), or (N,2) scores, got {y.shape}")
    elif y.ndim > 2:
        y = np.squeeze(y)
        if y.ndim > 1:
            raise ValueError(f"Expected (N,), (N,1), or (N,2) scores, got {y.shape}")

    y_score = y
    # logits? convert to probs
    if (y_score.min() < 0) or (y_score.max() > 1):
        y_score = _sigmoid(y_score)

    # Need both classes present
    if np.unique(y_true).size < 2:
        return None, None, y_score

    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = None
    try:
        pr = average_precision_score(y_true, y_score)
    except Exception:
        pr = None

    return roc, pr, y_score


def _save_cm_png(cm, class_names, fname, normalize=False, title=None):
    # Optionally normalize rows
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title or ('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix')
    )

    # Rotate x tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return fname


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

def build_backbone(cfg):
    backbone = cfg.get("backbone")
    n_classes = cfg.get("n_classes")
    if backbone == "MorphMIL":
        model = MorphMIL(cfg=cfg, n_classes=n_classes)
        return model
    raise ValueError(f"Unknown backbone: {backbone}")


def _unpack_out(out):
    if isinstance(out, dict):
        return out["logits"], out.get("Y_hat"), out.get("Y_prob")
    return out, None, None

# ---------- Trainer Class ----------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.is_ddp = bool(cfg.get("is_ddp", False))
        self.rank = int(cfg.get("rank", 0))
        self.world_size = int(cfg.get("world_size", 1))
        self.local_rank = int(cfg.get("local_rank", 0))

        # Seed: make it rank-dependent so workers don't all sample identically
        set_seed(cfg.get("seed", 42) + self.rank)

        # Device MUST be local_rank in DDP
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Data (keep your function)
        self.train_loader, self.val_loader, info = train_val_loaders(
            h5_dir=cfg.get("h5_dir"),
            morph_dir=cfg.get("morph_dir"),
            labels_csv=cfg.get("labels_csv"),
            val_ratio=0.2,
            seed=42,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            use_weighted_sampler=False,
        )

        # If DDP: replace train loader sampler with DistributedSampler
        if self.is_ddp:
            train_ds = self.train_loader.dataset
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False
            )
            self.train_loader = DataLoader(
                train_ds,
                batch_size=self.train_loader.batch_size,
                sampler=train_sampler,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory,
                drop_last=False,
            )
            # (Simplest) only rank0 runs val/test → keep val_loader as-is

        # Model
        self.model = build_backbone(cfg).to(self.device)

        # Wrap in DDP (after .to(device))
        if self.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=bool(cfg.get("find_unused_parameters", False)),
            )

        # Optimizer / Scheduler / Loss (same as you had)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("wd", 1e-4)),
        )
        self.scheduler, self._sched_mode = build_scheduler(self.optimizer, self.train_loader, cfg)

        weights = torch.tensor([1.0, 61/9], device=self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.use_amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Rank-0 only logging/checkpoints
        self._rank0 = (not self.is_ddp) or (self.rank == 0)

        self.outdir = cfg.get("outdir", "checkpoints")
        os.makedirs(self.outdir, exist_ok=True)
        self.ckpt_best = os.path.join(self.outdir, "best.pt")
        self.ckpt_last = os.path.join(self.outdir, "last.pt")
        self.best_val_acc = -1.0

        self.run_name = self.cfg.get("run_name", "exp")
        self.tb_dir = self.cfg.get("tb_dir", f"runs/{self.run_name}-{time.strftime('%Y%m%d-%H%M%S')}")
        self.tb = SummaryWriter(log_dir=self.tb_dir) if self._rank0 else None
        self.global_step = 0
        print("TensorBoard dir------------------------------:", self.tb_dir)

    # ---------- Train one epoch ----------
    @staticmethod
    def _unpack_out(out):
        if isinstance(out, dict):
            return out["logits"], out.get("Y_hat"), out.get("Y_prob")
        return out, None, None

    def _get_lr(self):
        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs

    def _current_lr(self):
        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs


    @torch.no_grad()
    def _predict_on_loader(self, loader):
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        for batch in loader:
            x = batch["feats"].to(self.device, non_blocking=True).unsqueeze(0)
            y = batch["label"].to(self.device, non_blocking=True)
            morph = batch["morph"].to(self.device, non_blocking=True).unsqueeze(0)               
            if self.use_amp:
                with autocast(enabled=self.use_amp):
                    out = self.model(x, morph)
            else:
                out = self.model(x, morph)

            logits, yhat, yprob = self._unpack_out(out)  # your existing helper
            print("logits, yhat, yprob--------------", logits.shape, yhat.shape, yprob.shape)
            preds = yhat if yhat is not None else logits.argmax(dim=1)
            probs = yprob if yprob is not None else torch.softmax(logits, dim=1)

            y_true.append(y.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(probs.detach().cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        y_prob = torch.cat(y_prob).numpy()
        return y_true, y_pred, y_prob    

    def test(self, loader=None, name="test"):
        """
        Evaluates on `loader` (defaults to self.test_loader if present, else self.val_loader).
        Prints accuracy, balanced accuracy, macro-F1, ROC-AUC, and saves confusion matrices.
        Returns (cm, report_dict).
        """
        import torch, random, numpy as np, os
        random.seed(123); np.random.seed(123); torch.manual_seed(123); torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)

        loader = loader or getattr(self, "test_loader", None) or self.val_loader

        # Expect these from your implementation
        y_true, y_pred, y_prob = self._predict_on_loader(loader)
        y_true_np = _to_numpy(y_true)
        y_pred_np = _to_numpy(y_pred)

        acc   = accuracy_score(y_true_np, y_pred_np) * 100.0
        bacc  = balanced_accuracy_score(y_true_np, y_pred_np) * 100.0
        mf1   = f1_score(y_true_np, y_pred_np, average="macro") * 100.0

        # Infer labels / class names if you wish; here binary example from your snippet:
        labels_numeric = sorted(np.unique(y_true_np).tolist())
        class_names = [str(c) for c in labels_numeric]  # or ["low","high"] etc.

        cm = confusion_matrix(y_true_np, y_pred_np, labels=labels_numeric)
        report = classification_report(y_true_np, y_pred_np, labels=labels_numeric,
                                    target_names=class_names, output_dict=True, digits=4)

        # ---- ROC-AUC (binary or multiclass)
        roc, pr, y_score = compute_binary_auc_and_ap(y_true_np, y_prob)
        roc_str = f"{roc:.3f}" if roc is not None else "NA"
        pr_str  = f"{pr:.3f}"  if pr  is not None else "NA"
        print(f"\n=== {name.upper()} RESULTS ===")
        print(f"acc={acc:.2f}%  bacc={bacc:.2f}%  macroF1={mf1:.2f}%  ROC-AUC={roc_str}  PR-AUC={pr_str}")
        print("Confusion matrix (rows=true, cols=pred):")


        with np.errstate(all='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        print("Confusion matrix (normalized):")
        np.set_printoptions(precision=3, suppress=True)
        print(cm_norm)

        # ---- Save PNGs (your existing helper)
        out_dir = getattr(self, "pics", "pics")
        os.makedirs(out_dir, exist_ok=True)
        if self.cfg["backbone"]=="TransMIL":
            raw_path  = os.path.join(out_dir, f"{name}_{self.cfg['backbone']}_cm.png")
            norm_path = os.path.join(out_dir, f"{name}_{self.cfg['backbone']}_cm_norm.png")
        elif self.cfg["backbone"]=="poolMIL":
            raw_path  = os.path.join(out_dir, f"{name}_{self.cfg['backbone']}_{self.cfg['pool']}_cm.png")
            norm_path = os.path.join(out_dir, f"{name}_{self.cfg['backbone']}_{self.cfg['pool']}_cm_norm.png")
            
        _save_cm_png(cm, class_names, raw_path,  normalize=False, title=f"{name} • Confusion Matrix")
        _save_cm_png(cm, class_names, norm_path, normalize=True,  title=f"{name} • Confusion Matrix (row-norm)")
        print(f"Saved confusion matrices to:\n  {raw_path}\n  {norm_path}")

        # ---- Log to TensorBoard
        if hasattr(self, "tb"):
            self.tb.add_scalar(f"{name}/acc",     acc,  0)
            self.tb.add_scalar(f"{name}/bacc",    bacc, 0)
            self.tb.add_scalar(f"{name}/macroF1", mf1,  0)

            # use roc/pr, not 'auc'
            if roc is not None:
                self.tb.add_scalar(f"{name}/ROC_AUC", roc, 0)
            if pr  is not None:
                self.tb.add_scalar(f"{name}/PR_AUC",  pr,  0)

            # Pretty PR curve (requires y_score in [0,1])
            try:
                self.tb.add_pr_curve(f"{name}/PR_curve", y_true_np.astype(int), y_score, global_step=0)
            except Exception as e:
                print("TensorBoard PR curve logging failed:", e)
                
        return cm, report
    # ---------- Train one epoch ----------
    def train_one_epoch(self, epoch: int):
        if self.is_ddp and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)
            
        self.model.train()
        running_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(self.train_loader, 1):
            x = batch["feats"].to(self.device, non_blocking=True).unsqueeze(0)   # (B,16,2048)
            y = batch["label"].to(self.device, non_blocking=True)   # (B,)
            morph = batch["morph"].to(self.device, non_blocking=True).unsqueeze(0)               

            i = 0

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                out = self.model(x, morph)                        # dict or tensor
                logits, yhat, yprob = self._unpack_out(out)
                # print("logits------------------------", logits.shape)
                # print("yhat--------------------------", yhat.shape, yhat)
                # print("yprob-------------------------", yprob.shape, yprob)
                # print("y-----------------------------", y.shape)
                # exit()
                loss = self.loss_fn(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            running_loss += loss.item() * x.size(0)
            preds = yhat if yhat is not None else logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total   += y.numel()

            # ---- TensorBoard: step logs (rank-0 only)
            if self._rank0 and self.tb is not None:
                # step-wise (use batch accuracy so it’s not noisy from running totals)
                batch_acc = (preds == y).float().mean().item() * 100.0
                self.tb.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.tb.add_scalar("train/acc_step",  batch_acc,  self.global_step)
                self.tb.add_scalar("train/lr",        self._get_lr(), self.global_step)
            self.global_step += 1


            if step % self.cfg.get("log_every", 50) == 0:
                avg_loss = running_loss / max(1, n_total)
                acc = 100.0 * n_correct / max(1, n_total)
                print(f"[Epoch {epoch} | Step {step}] loss={avg_loss:.4f} acc={acc:.2f}%")

        epoch_loss = running_loss / max(1, n_total)
        epoch_acc  = 100.0 * n_correct / max(1, n_total)
        
        # ---- TensorBoard: epoch logs
        if self._rank0 and self.tb is not None:
            self.tb.add_scalar("train/epoch_loss", epoch_loss, epoch)
            self.tb.add_scalar("train/epoch_acc",  epoch_acc,  epoch)        
            
        
        return epoch_loss, epoch_acc

    # ---------- Validation ----------
    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        running_loss, n_correct, n_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["feats"].to(self.device, non_blocking=True).unsqueeze(0)
                y = batch["label"].to(self.device, non_blocking=True)
                morph = batch["morph"].to(self .device, non_blocking=True).unsqueeze(0)               

                if self.use_amp:
                    with autocast(enabled=self.use_amp):
                        out = self.model(x, morph)
                        logits, yhat, yprob = self._unpack_out(out)
                        loss = self.loss_fn(logits, y)
                else:
                    out = self.model(x, morph)
                    logits, yhat, yprob = self._unpack_out(out)
                    loss = self.loss_fn(logits, y)

                running_loss += loss.item() * x.size(0)
                preds = yhat if yhat is not None else logits.argmax(dim=1)
                n_correct += (preds == y).sum().item()
                n_total   += y.numel()

        val_loss = running_loss / max(1, n_total)
        val_acc  = 100.0 * n_correct / max(1, n_total)


        # ---- TensorBoard: epoch logs
        if self._rank0 and self.tb is not None:
            self.tb.add_scalar("val/loss", val_loss, epoch)
            self.tb.add_scalar("val/acc",  val_acc,  epoch)

        # Step ReduceLROnPlateau (metric mode) here
        # if self.scheduler and self._sched_mode == "metric":
        #     metric_name = self.cfg.get("plateau_metric", "loss")  # "loss" | "acc"
        #     metric = val_loss if metric_name == "loss" else val_acc
        #     self.scheduler.step(metric)
        lr = self._get_lr()
        return val_loss, val_acc, lr

    # ---------- Save / Load ----------
    def save_ckpt(self, path, epoch, val_acc):
        torch.save({
            "cfg": self.cfg,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, path)

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt

    # ---------- Full fit ----------
    def fit(self):
        mode = self.cfg.get("mode", "train")

        # -------------------------
        # TRAIN
        # -------------------------
        if mode == "train":
            epochs = int(self.cfg.get("epochs", 20))

            for epoch in range(1, epochs + 1):
                # everyone trains
                tr_loss, tr_acc = self.train_one_epoch(epoch)

                # rank0 validates / logs / checkpoints
                if self._rank0:
                    val_loss, val_acc, lr = self.validate(epoch)

                    print(
                        f"Epoch {epoch:02d} | "
                        f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% lr={lr:.6f}"
                    )

                    # save last
                    self.save_ckpt(self.ckpt_last, epoch, val_acc)

                    # save best
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.save_ckpt(self.ckpt_best, epoch, val_acc)
                        print(f"🔥 New best val_acc: {val_acc:.2f}% — saved to {self.ckpt_best}")

                # keep all ranks in sync (so non-rank0 doesn't run ahead)
                if self.is_ddp:
                    dist.barrier()

            # Final eval only on rank0
            if self._rank0:
                self.load_ckpt(self.ckpt_best)
                print("Loaded best checkpoint for final evaluation.")
                self.test(loader=getattr(self, "test_loader", None), name="test")

                # TensorBoard clean up
                if self.tb is not None:
                    self.tb.flush()
                    self.tb.close()

            if self.is_ddp:
                dist.barrier()

            return

        # -------------------------
        # TEST ONLY
        # -------------------------
        elif mode == "test":
            if self._rank0:
                print("self.ckpt_best-------------", self.ckpt_best)
                self.load_ckpt(self.ckpt_best)
                print("Loaded best checkpoint for final evaluation.")
                self.test(loader=getattr(self, "test_loader", None), name="test")

            if self.is_ddp:
                dist.barrier()

            return

        else:
            raise ValueError(f"Unknown mode={mode}. Expected 'train' or 'test'.")
        
        
        




# ---------- Main ----------
def main():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/trainMorphMIL.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    is_ddp, rank, world_size, local_rank = ddp_setup()
    cfg["is_ddp"] = is_ddp
    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["local_rank"] = local_rank

    trainer = Trainer(cfg)
    trainer.fit()

    ddp_cleanup()

if __name__ == "__main__":
    main()
