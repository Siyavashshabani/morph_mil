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
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.utils.class_weight import compute_sample_weight
# import timm
from model.morphMIL import MorphMIL
from model.poolMIL import PoolMIL
from model.utils import build_scheduler, build_loss
import os, time
from pathlib import Path

import torch
import torch.nn.functional as F
from loss.contrastiveLoss import SupConLoss

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
        device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")
        model = MorphMIL(cfg= cfg, n_classes = n_classes).to(device)
        return model


def _unpack_out(out):
    if isinstance(out, dict):
        return out["logits"], out.get("Y_hat"), out.get("Y_prob")
    return out, None, None

# ---------- Trainer Class ----------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg.get("seed", 42))
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gpu_id = cfg.get("cuda", 0)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        # Data 
        self.dataset = cfg.get("dataset")
        if self.dataset=="brca":  
            from dataloader.dataloader import train_val_loaders   
            self.train_loader, self.val_loader, info = train_val_loaders(
                h5_dir=cfg.get("h5_dir"),
                morph_dir=cfg.get("morph_dir"),
                labels_csv=cfg.get("labels_csv"),
                val_ratio=0.2,
                seed=42,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                use_weighted_sampler= False, #True
                aug_flag=cfg.get("aug_flag")
            )
        elif self.dataset=="camelyon":
            print("camelyon-----------------------------------------------")
            from dataloader.dataloaderCamelyon import train_val_loaders 
            self.train_loader, self.val_loader, self.test_loader = train_val_loaders(
                h5_dir=cfg.get("h5_dir"),
                morph_dir=cfg.get("morph_dir"),
                labels_csv=cfg.get("labels_csv"),
                val_ratio=0.2,
                seed=42,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                use_weighted_sampler= False, #True
                aug_flag=cfg.get("aug_flag")
            )


        # Model / Optim / Loss
        base_dim = self.cfg.get("base_input_dim", 1024)
        morph_dim = self.cfg.get("morph_dim", 246)
        
        # if self.cfg.get("simple_concat", False):
        #     self.cfg["input_dim"] = base_dim + morph_dim
        # else:
        #     self.cfg["input_dim"] = base_dim        
                    
        self.model = build_backbone(cfg).to(self.device)

        ## define the optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("wd", 1e-4)),
        )
        self.scheduler, self._sched_mode = build_scheduler(self.optimizer, self.train_loader, cfg)
        
        ## define the lossfunction
        # weights = torch.tensor([1.0, 61/9], device=self.device )  # use your train counts
        # self.loss_fn = nn.CrossEntropyLoss(weight=weights )

        # AMP
        self.use_amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # ---- run id / folder name ----
        self.run_name = self.cfg.get("run_name", "exp")
        run_id = self.cfg.get("run_id", time.strftime("%Y%m%d-%H%M%S"))  # or uuid4()

        # root folder for everything in this run
        self.run_dir = Path(self.cfg.get("run_root", "experiments")) / f"{self.run_name}-{run_id}"

        # only rank0 creates dirs and writes ckpts/tb
        self._rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if self._rank0:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        # ---- checkpoints ----
        self.outdir = str(self.run_dir / "checkpoints")
        if self._rank0:
            os.makedirs(self.outdir, exist_ok=True)

        self.ckpt_best = os.path.join(self.outdir, "best.pt")
        self.ckpt_last = os.path.join(self.outdir, "last.pt")
        self.best_val_acc = -1.0

        # ---- tensorboard ----
        self.tb_dir = str(self.run_dir / "tb")
        self.tb = SummaryWriter(log_dir=self.tb_dir) if self._rank0 else None

        self.global_step = 0   # increments each train step

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


    def augmix_jsd_loss_from_probs(
        self,
        feats: torch.Tensor,          # e.g. [3, N, D] (3 views, N tokens, D dim)
        yprob: torch.Tensor,          # [3, C] probabilities
        target: torch.Tensor,         # scalar or [1]
        alpha_jsd: float = 50.0,
        alpha_con: float = 50.0,
        eps: float = 1e-7,
        return_parts: bool = True,
        loss_type: str = "ce_jsd",    # "ce", "jsd", "ce_jsd", "ce_con"
        temperature: float = 0.1,
    ):
        # -------------------------
        # checks + canonical dtypes
        # -------------------------
        if yprob.dim() != 2 or yprob.size(0) != 3:
            raise ValueError(f"Expected yprob [3, C], got {tuple(yprob.shape)}")

        # probs to float32 for stability
        yprob = yprob.float()

        # target -> [1] long on same device
        target = torch.as_tensor(target, device=yprob.device, dtype=torch.long)
        if target.dim() == 0:
            target = target.view(1)
        if target.dim() != 1 or target.numel() != 1:
            raise ValueError(f"target must be scalar or [1], got {tuple(target.shape)}")

        # --------------------------------
        # split probabilities: [1, C] each
        # --------------------------------
        p_clean = yprob[0:1]
        p_aug1  = yprob[1:2]
        p_aug2  = yprob[2:3]
        
        ## making ram free
        del yprob
        
        # -------------------------
        # CE (on clean) from probs
        # -------------------------
        p_clean_log = torch.log(torch.clamp(p_clean, eps, 1.0))
        ce = F.nll_loss(p_clean_log, target)

        # -------------------------
        # JSD (AugMix style)
        # -------------------------
        p_mix = (p_clean + p_aug1 + p_aug2) / 3.0
        p_mix_log = torch.log(torch.clamp(p_mix, eps, 1.0))

        # IMPORTANT: F.kl_div expects (input=log-probs, target=probs) if log_target=False
        kl_clean = F.kl_div(p_mix_log, p_clean, reduction="batchmean")
        kl_aug1  = F.kl_div(p_mix_log, p_aug1,  reduction="batchmean")
        kl_aug2  = F.kl_div(p_mix_log, p_aug2,  reduction="batchmean")
        jsd = (kl_clean + kl_aug1 + kl_aug2) / 3.0

        # -------------------------
        # SupCon (3 views)
        # feats expected like [3, N, D] or [3, D]
        # We pool tokens -> [3, D] then -> [1, 3, D]
        # labels should be [bsz]=[1]
        # -------------------------
        con = torch.tensor(0.0, device=self.device)
        if loss_type == "ce_con" or return_parts:
            if feats is None:
                raise ValueError("feats is required for contrastive loss (ce_con).")

            if feats.dim() == 3 and feats.size(0) == 3:
                # [3, N, D] -> [3, D]
                feats_view = feats.mean(dim=1)
            elif feats.dim() == 2 and feats.size(0) == 3:
                # [3, D]
                feats_view = feats
            else:
                raise ValueError(f"Expected feats [3,N,D] or [3,D], got {tuple(feats.shape)}")

            feats_view = F.normalize(feats_view, dim=-1)
            feats_sc = feats_view.unsqueeze(0)  # [1, 3, D]

            con_loss_fn = SupConLoss(temperature=temperature, contrast_mode="all").to(self.device)
            con = con_loss_fn(feats_sc, labels=target)

        # -------------------------
        # choose loss
        # -------------------------
        if loss_type == "ce":
            loss = ce
        elif loss_type == "jsd":
            loss = jsd
        elif loss_type == "ce_jsd":
            loss = ce + alpha_jsd * jsd
        elif loss_type == "ce_con":
            loss = ce + alpha_con * con
        else:
            raise ValueError(f"Unknown loss_type='{loss_type}'. Use: ce, jsd, ce_jsd, ce_con")

        if return_parts:
            return {
                "loss": loss,
                "ce": ce.detach(),
                "jsd": jsd.detach(),
                "con": con.detach(),
            }
        return loss
            
        # return loss, {"ce": ce, "jsd": jsd, "kl_clean": kl_clean, "kl_aug1": kl_aug1, "kl_aug2": kl_aug2}




    @torch.no_grad()
    def _predict_on_loader(self, loader):
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        for batch in loader:
            x = batch["feats"].to(self.device, non_blocking=True) #.unsqueeze(0)
            y = batch["label"].to(self.device, non_blocking=True)
            morph = batch["morph"].to(self.device, non_blocking=True).unsqueeze(0)               
            if self.use_amp:
                with autocast():
                    out = self.model(x, morph)
            else:
                out = self.model(x, morph)

            logits, yhat, yprob = self._unpack_out(out)  # your existing helper
            print("logits, yhat, yprob--------------", logits.shape, yhat.shape, yprob.shape)
            preds = yhat[0:1] if yhat[0:1] is not None else logits[0:1].argmax(dim=1)
            probs = yprob[0:1] if yprob[0:1] is not None else torch.softmax(logits[0:1], dim=1)

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

        loader = loader or getattr(self, "test_loader", None) or self.test_loader

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

        elif self.cfg["backbone"]=="MorphMIL":
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
        self.model.train()
        running_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(self.train_loader, 1):
            x = batch["feats"].to(self.device, non_blocking=True)   # (B,16,2048)
            # print("x.shape-----------------------", x.shape)
            target = batch["label"].to(self.device, non_blocking=True)  # (B,)
            morph = batch["morph"].to(self.device, non_blocking=True).unsqueeze(0)               
            i = 0

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = self.model(x, morph)                        # dict or tensor
                logits, yhat, yprob = self._unpack_out(out)
                loss = self.augmix_jsd_loss_from_probs(x, yprob, target, 
                                                       return_parts=False, 
                                                       loss_type= self.cfg.get("loss_type"),      
                                                       alpha_jsd=self.cfg.get("alpha_jsd"),
                                                       alpha_con=self.cfg.get("alpha_con")
                                                       )
                # print("loss_total:", loss.item())
                # exit()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            running_loss += loss.item() * x.size(0)
            preds = yhat[0:1] if yhat[0:1] is not None else logits[0:1].argmax(dim=1)
            n_correct += (preds == target).sum().item()
            n_total   += target.numel()

            # ---- TensorBoard: step logs (rank-0 only)
            if self._rank0 and self.tb is not None:
                # step-wise (use batch accuracy so it’s not noisy from running totals)
                batch_acc = (preds == target).float().mean().item() * 100.0
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
                x = batch["feats"].to(self.device, non_blocking=True)
                target = batch["label"].to(self.device, non_blocking=True)
                morph = batch["morph"].to(self .device, non_blocking=True).unsqueeze(0)               

                if self.use_amp:
                    with autocast():
                        out = self.model(x, morph)
                        logits, yhat, yprob = self._unpack_out(out)
                        loss = self.augmix_jsd_loss_from_probs(x, yprob, target, 
                                                               return_parts=False, 
                                                               loss_type= self.cfg.get("loss_type"),
                                                               alpha_jsd=self.cfg.get("alpha_jsd"),
                                                               alpha_con=self.cfg.get("alpha_con")
                                                               )
                else:
                    out = self.model(x, morph)
                    logits, yhat, yprob = self._unpack_out(out)
                    loss = self.augmix_jsd_loss_from_probs(x, yprob, target,
                                                           eturn_parts=False, 
                                                           loss_type= self.cfg.get("loss_type"),
                                                           alpha_jsd=self.cfg.get("alpha_jsd"),
                                                           alpha_con=self.cfg.get("alpha_con")
                                                           )

                running_loss += loss.item() * x.size(0)
                preds = yhat[0:1] if yhat[0:1] is not None else logits[0:1].argmax(dim=1)
                n_correct += (preds == target).sum().item()
                n_total   += target.numel()
                
                ## make free the memory               
                del out, logits, yhat, yprob, loss, x, target, morph, preds

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
        if self.cfg["mode"]=="train": 
            epochs = int(self.cfg.get("epochs", 20))
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = self.train_one_epoch(epoch)
                val_loss, val_acc, lr = self.validate(epoch)

                print(f"Epoch {epoch:02d} | "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% lr={lr:.6f}")

                # save last
                self.save_ckpt(self.ckpt_last, epoch, val_acc)

                # save best
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_ckpt(self.ckpt_best, epoch, val_acc)
                    print(f"🔥 New best val_acc: {val_acc:.2f}% — saved to {self.ckpt_best}")

            self.load_ckpt(self.ckpt_best)   # implement this if you haven't
            print("Loaded best checkpoint for final evaluation.")
            self.test(loader=getattr(self, "test_loader", None), name="test")

            # ---- TensorBoard clean up
            if self._rank0 and self.tb is not None:
                self.tb.flush()
                self.tb.close()
        
        elif self.cfg["mode"]=="test": 
            print("self.ckpt_best-------------", self.ckpt_best)
            self.load_ckpt(self.ckpt_best)   # implement this if you haven't
            print("Loaded best checkpoint for final evaluation.")
            self.test(loader=getattr(self, "test_loader", None), name="test")
        
        
        

# ---------- Main ----------
def main():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/trainJsdMorphMIL.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    trainer = Trainer(cfg)

    # quick check one batch
    # trainer.forward_all(ckpt_path=None, max_batches=1)
    # print("trainer.forward_all-------------------------pass")
    # full training        
    trainer.fit()
    print(cfg)

if __name__ == "__main__":
    main()
