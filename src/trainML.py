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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from pathlib import Path
from datetime import datetime
import timm
from model.autoencoder import Autoencoder, _load_state
from model.autoencoderResNetFull import ResFullAutoencoder

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

def build_backbone(cfg):
    backbone = cfg.get("backbone", "resnet18")

    name = cfg.get("backbone", "resnet18")
    features_only = cfg.get("features_only", False)         # set True for segmentation
    out_indices = tuple(cfg.get("out_indices", (1, 2, 3, 4)))  # stages to return if features_only
    global_pool = cfg.get("global_pool", "avg")             # "avg" / "max" / "catavgmax" etc.
    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")

    if backbone == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1")
        feat_dim = model.fc.in_features  # 512
        model.fc = nn.Identity()
        print(f"Using ResNet18, feature dim = {feat_dim}")

    elif backbone == "resnet50":
        model = resnet50(weights="IMAGENET1K_V1")
        feat_dim = model.fc.in_features  # 2048
        model.fc = nn.Identity()
        print(f"Using ResNet50, feature dim = {feat_dim}")

    elif backbone == "resnet101":
        model = resnet101(weights="IMAGENET1K_V1")
        feat_dim = model.fc.in_features  # 2048
        model.fc = nn.Identity()
        print(f"Using ResNet101, feature dim = {feat_dim}")

    # EfficientViT via timm (e.g., "efficientvit_b1.r224_in1k", "efficientvit_b2.r288_in1k", "efficientvit_m5.r224_in1k")
    elif name.startswith("efficientvit"):
        # Allow sloppy names and map to a sane default
        if name in {"efficientvit", "efficientvit_"}:
            name = cfg.get("efficientvit_name", "efficientvit_b2.r288_in1k")

        try:
            if features_only:
                model = timm.create_model(
                    name, pretrained=True, features_only=True, out_indices=out_indices
                )
            else:
                model = timm.create_model(name, pretrained=True)
                # make it a pure feature extractor
                model.reset_classifier(num_classes=0, global_pool=global_pool)
            print(f"Using {name} (timm {timm.__version__})")
            return model

        except RuntimeError as e:
            if "Unknown model" in str(e):
                avail = timm.list_models("efficientvit*")
                raise ValueError(
                    f"Unknown EfficientViT model '{name}'. "
                    f"Installed timm={timm.__version__}. "
                    f"Available: {avail}"
                )
            raise
        
    elif backbone == "autoencoder":
        in_ch  = int(cfg.get("in_channels", 1))
        out_ch = int(cfg.get("out_channels", in_ch))
        out_act = cfg.get("out_activation", "sigmoid")

        ae = Autoencoder(
            in_channels=in_ch,
            out_channels=out_ch,
            enc_channels=(64, 128, 256, 512, 1024),
            latent_dim=2048,
            base_grid=7,
            decoder_widths=(512, 256, 128, 64, 32),
            out_activation=out_act,
        )

        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            print("check is available--------------------------")
            state, epoch = _load_state(ckpt_path, device)
            ae.load_state_dict(state, strict=True)
            print(f"[load] {ckpt_path} (epoch={epoch})")

        ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad = False

        model = ae.eval()  # model(x) -> (B, 2048)
        return model

    elif backbone == "resautoencoder":
        in_ch  = cfg.get("in_channels")
        out_ch = cfg.get("out_channels")
        out_act = cfg.get("out_activation")
        enc_out = cfg.get("encoder_out")
        # 1) Build the full AE (encoder+decoder) so the state_dict matches
        ae = ResFullAutoencoder(in_ch=in_ch, 
                                out_ch=out_ch, 
                                final_activation=out_act, 
                                encoder_out=enc_out)
        ckpt_path = cfg.get("ckpt_path")

        if ckpt_path:
            state, epoch = _load_state(ckpt_path, device)
            # If your ckpt was saved with DDP, strip 'module.' if needed:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            ae.load_state_dict(state, strict=True)
            print(f"[load] {ckpt_path} (epoch={epoch})")

        ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad = False

        # 2) Wrap ONLY the encoder with avgpool+flatten to get a 2048-D vector
        if cfg["encoder_out"]=="map":
            feature_model = nn.Sequential(
                ae.encoder,                  # your ResEncoder -> [B, 2048, H/32, W/32]
                nn.AdaptiveAvgPool2d(1),     # -> [B, 2048, 1, 1]
                nn.Flatten(start_dim=1)      # -> [B, 2048]
            ).to(device).eval()

        if cfg["encoder_out"]=="vector":
            feature_model = nn.Sequential(
                ae.encoder                  # your ResEncoder -> [B, 2048, H/32, W/32]
                # nn.AdaptiveAvgPool2d(1),     # -> [B, 2048, 1, 1]
                # nn.Flatten(start_dim=1)      # -> [B, 2048]
            ).to(device).eval()


        # (optional) double-freeze the wrapper (already frozen above, but explicit is fine)
        for p in feature_model.parameters():
            p.requires_grad = False

        print("Using Autoencoder encoder, feature dim = 2048")
        return feature_model   # model(x) -> [B, 2048]



    else:
        raise ValueError(f"Unknown backbone: {backbone}")




    return model


# ---------- Trainer Class ----------
class Trainer:
    def __init__(self, cfg):
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
            batch_size=cfg.get("batch_size", 1),
            num_workers=cfg.get("num_workers", 2),
            seed=cfg.get("seed", 42),
        )

        # Model / Optim / Loss
        self.model = build_backbone(cfg).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=cfg.get("lr", 1e-3),
                                     weight_decay=cfg.get("wd", 1e-4))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Checkpoints
        self.outdir = cfg.get("outdir", "checkpoints")
        self.ckpt_best = os.path.join(self.outdir, "best.pt")
        self.ckpt_last = os.path.join(self.outdir, "last.pt")

        self.best_val_acc = -1.0


    def sparse_select(self, X, y, cv=5):
        # Scale features (important for LASSO)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Fit LASSO with cross-validation
        lasso = LassoCV(cv=cv, random_state=0)
        lasso.fit(Xs, y)

        # Get selected features
        selected = np.where(lasso.coef_ != 0)[0]
        X_reduced = X[:, selected]

        return X_reduced, selected  
          
    @torch.no_grad()
    def compute_aggregated_embeddings(self, split="val", max_batches=None):
        """
        Run the model on the given split and collect mean+median logits with paths.

        Args:
            split (str): "train" or "val"
            max_batches (int, optional): limit the number of batches processed.

        Returns:
            all_logits (torch.Tensor): shape [num_samples, 2*num_classes]
            all_paths (list of str): list of file paths for each sample
        """
        self.model.eval()
        loader = self.train_loader if split == "train" else self.val_loader
        all_logits, all_paths, all_labels = [], [], []

        for i, batch in enumerate(loader):
            x = batch["image"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if self.cfg["backbone"]=="autoencoder":
                    _,output = self.model(x)  
                else: 
                    output = self.model(x) 

                

                if self.cfg["merge_mode"] == "mean_median":
                    # mean [1, num_classes]
                    output_mean = output.mean(dim=0, keepdim=True)
                    # median returns namedtuple
                    median_result = output.median(dim=0, keepdim=True)
                    output_median = median_result.values  # [1, num_classes]
                    merged_vector = torch.cat([output_mean, output_median], dim=1)
                    
                elif self.cfg["merge_mode"] == "mean":   
                    # mean [1, num_classes]
                    output_mean = output.mean(dim=0, keepdim=True)
                    merged_vector = output_mean

                elif self.cfg["merge_mode"] == "max":   
                    # max 
                    output_max, _ = output.max(dim=0, keepdim=True)
                    merged_vector = output_max
                

                # print("shape of mean_median-------------------", merged_vector.shape)
            all_logits.append(merged_vector.cpu())
            all_paths.append(batch["path"][0])
            all_labels.append(batch["tertile_str"][0])

        # stack to [num_samples, 2*num_classes]
        all_logits = torch.cat(all_logits, dim=0)
        
        ## spase regression for decreasing the length of signal                
        # print("all_logits-------------------------------------:", all_logits.shape)
        # print("batch['tertile_str'][0]------------------------:", len(all_labels),all_labels[0] )        
        # X_reduced, selected = self.sparse_select(merged_vector, all_labels)
        # exit()
        
                
        
        
        return all_logits, all_paths, all_labels

    @torch.no_grad()
    def forward_all(
        self, ckpt_path: str = None, max_batches: int = None
    ):
        """
        Run the model forward on all images in train/val dataloader and return aggregated embeddings.

        Args:
            split (str): 'train' or 'val'
            ckpt_path (str, optional): optional path to a checkpoint to load before forwarding
            max_batches (int, optional): if set, only process this many batches

        Returns:
            all_logits (torch.Tensor): concatenated tensor of embeddings [N, 2*num_classes]
            all_paths (list[str]): list of image paths
        """
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)  # make sure you have this function in your class
            
        ################--------------train--------------################
        split = "train"
        x_train, path_train, y_train = self.compute_aggregated_embeddings(split=split, max_batches=max_batches)
        print(f"[forward_all] {split}: {x_train.shape}, {len(path_train)} samples")

        ################--------------validation--------------################
        split = "val"
        x_val, path_val, y_val = self.compute_aggregated_embeddings(split=split, max_batches=max_batches)
        print(f"[forward_all] {split}: {x_val.shape}, {len(path_val)} samples")

        # ---- Convert features to numpy
        X_train = x_train.detach().cpu().numpy() if torch.is_tensor(x_train) else np.asarray(x_train)
        X_val   = x_val.detach().cpu().numpy()   if torch.is_tensor(x_val)   else np.asarray(x_val)

        # ---- Encode labels (handles string or int labels)
        y_train = np.asarray(y_train)
        y_val   = np.asarray(y_val)
        if y_train.dtype.kind in {"U", "S", "O"}:  # strings/objects → encode
            le = LabelEncoder().fit(y_train)       # learns ["high","low","medium"] sorted lexicographically
            y_train_enc = le.transform(y_train)
            y_val_enc   = le.transform(y_val)
            class_names = list(le.classes_)
        else:
            # already numeric; ensure 0..K-1 and set names
            y_train_enc = y_train.astype(int)
            y_val_enc   = y_val.astype(int)
            class_names = ["low", "mid", "high"]  # adjust if your mapping differs

        # ---- Candidate models

        def _to_float32(X):
            # works for numpy arrays or pandas DataFrames
            try:
                X = X.to_numpy()
            except AttributeError:
                pass
            return np.asarray(X, dtype=np.float32)

        to32 = FunctionTransformer(_to_float32, validate=False)

        models = {
            
            ## random forest 
            "rf": Pipeline([ ("scaler", StandardScaler()), 
                    ("clf", RandomForestClassifier(n_estimators=400, 
                    class_weight="balanced_subsample", 
                    n_jobs=-1, 
                    random_state=42)) ]),            
            ## SVM
            "linear_svm": Pipeline([ ("scaler", StandardScaler()), 
                    ("clf", LinearSVC(class_weight="balanced")) ]),            
            
            # KNN
            "knn": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))
            ]),
            # Decision Tree
            "dt": Pipeline([
                ("scaler", "passthrough"),
                ("clf", DecisionTreeClassifier(
                    max_depth=None, min_samples_leaf=5,
                    class_weight="balanced", random_state=42
                ))
            ]),
            # AdaBoost
            "ada": Pipeline([
                ("scaler", "passthrough"),
                ("clf", AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=2, class_weight="balanced", random_state=42
                    ),
                    n_estimators=300, learning_rate=0.05, random_state=42
                ))
            ]),
            # Logistic Regression (L1)
            "logreg_l1": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l1", solver="saga", max_iter=5000,
                    class_weight="balanced", random_state=42
                ))
            ]),
            # === Newly added ===
            "lda": Pipeline([
                    ("to32", to32),
                    ("scaler", "passthrough"),
                    ("clf", LinearDiscriminantAnalysis())
                ]),
            "qda": Pipeline([
                    ("to32", to32),
                    ("scaler", "passthrough"),
                    ("clf", QuadraticDiscriminantAnalysis(reg_param=0.0))
                ]),
            # "mlp": Pipeline([
            #     ("scaler", StandardScaler()),
            #     ("clf", MLPClassifier(
            #         hidden_layer_sizes=(128, 64), activation="relu",
            #         alpha=1e-4, learning_rate_init=1e-3,
            #         max_iter=300, early_stopping=True, random_state=42
            #     ))
            # ]),
        }





        # # ---- 5-fold CV on train (macro-F1) to pick model
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cv_scores = {}
        # for name, pipe in models.items():
        #     scores = cross_val_score(pipe, X_train, y_train_enc, cv=cv, scoring="f1_macro", n_jobs=-1)
        #     cv_scores[name] = (scores.mean(), scores.std())

        # print("CV macro-F1:")
        # for k, (m, s) in cv_scores.items():
        #     print(f"{k:>10s}: {m:.3f} ± {s:.3f}")

        # best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
        # best_model = models[best_name].fit(X_train, y_train_enc)

        # # ---- Evaluate on held-out val
        # y_pred = best_model.predict(X_val)

        # acc  = accuracy_score(y_val_enc, y_pred)
        # bacc = balanced_accuracy_score(y_val_enc, y_pred)
        # mf1  = f1_score(y_val_enc, y_pred, average="macro")
        # print(f"\nHeld-out performance:")
        # print(f"Accuracy          : {acc:.3f}")
        # print(f"Balanced Accuracy : {bacc:.3f}")
        # print(f"Macro-F1          : {mf1:.3f}")

        # # Ensure consistent label order in the report/confusion matrix
        # labels_order = sorted(np.unique(y_train_enc))  # typically [0,1,2]
        # print("\nClassification report:\n",
        #     classification_report(y_val_enc, y_pred,
        #                             labels=labels_order,
        #                             target_names=class_names))



        # Fit each model and evaluate on the validation set
        val_metrics = {}
        for name, pipe in models.items():
            m = pipe.fit(X_train, y_train_enc)
            y_pred = m.predict(X_val)

            # ---- scores for ROC–AUC ----
            if hasattr(m, "predict_proba"):
                y_score = m.predict_proba(X_val)[:, 1]
            elif hasattr(m, "decision_function"):
                y_score = m.decision_function(X_val)   # works for e.g. linear SVM
            else:
                y_score = None  # can't compute AUC for this model

            # ---- metrics ----
            acc  = accuracy_score(y_val_enc, y_pred)
            bacc = balanced_accuracy_score(y_val_enc, y_pred)
            mf1  = f1_score(y_val_enc, y_pred, average="macro")

            if (y_score is not None) and (np.unique(y_val_enc).size == 2):
                auc = roc_auc_score(y_val_enc, y_score)
            else:
                auc = float("nan")

            val_metrics[name] = {"acc": acc, "bacc": bacc, "mf1": mf1, "roc_auc": auc, "model": m}

        # Show a sorted summary (by AUC first, then accuracy)
        rows = sorted(
            [(n, d["roc_auc"], d["acc"], d["bacc"], d["mf1"]) for n, d in val_metrics.items()],
            key=lambda r: (-(r[1] if np.isfinite(r[1]) else -1), -r[2])
        )
        print("\nValidation results (sorted by ROC-AUC then accuracy):")
        for n, auc, acc, bacc, mf1 in rows:
            print(f"{n:>12s}  auc={auc:.3f}  acc={acc:.3f}  bacc={bacc:.3f}  mf1={mf1:.3f}")
        # Pick the best by accuracy
        best_by_acc = max(val_metrics, key=lambda k: val_metrics[k]["acc"])
        best_model  = val_metrics[best_by_acc]["model"]
        print(f"\nBest by accuracy: {best_by_acc} (acc={val_metrics[best_by_acc]['acc']:.3f})")


        # predict with the best model
        y_pred_best = best_model.predict(X_val)

        # consistent label order
        labels_order = sorted(np.unique(y_train_enc))

        # choose display names
        try:
            display_names = list(le.classes_)              # if you used LabelEncoder
        except NameError:
            try:
                display_names = class_names                # if you defined ["low","medium","high"]
            except NameError:
                display_names = [str(i) for i in labels_order]

        # compute confusion matrix
        cm = confusion_matrix(y_val_enc, y_pred_best, labels=labels_order)

        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

        # (optional) pretty print as a table
        try:
            import pandas as pd
            cm_df = pd.DataFrame(cm,
                                index=[f"true_{n}" for n in display_names],
                                columns=[f"pred_{n}" for n in display_names])
            print("\nConfusion matrix (labeled):")
            print(cm_df)
        except ImportError:
            pass

        # (optional) normalized by true rows
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print("\nRow-normalized confusion matrix:")
        print(cm_norm)

        ####################################### save the .csv resutls 
        # rows is: [(name, auc, acc, bacc, mf1), ...]
        df = pd.DataFrame(rows, columns=["model", "roc_auc", "acc", "bacc", "mf1"])

        # (optional) keep full precision in file; round only if you want pretty numbers
        # df = df.round(3)

        # choose where to save
        results_dir = Path("results/trainML")       # or Path(out_dir) / "metrics"
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"val_results_{self.cfg['merge_mode']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

        df.to_csv(csv_path, index=False)
        print(f"Saved metrics to: {csv_path}")        
                



        return x_train, path_train




# ---------- Main ----------
def main():
    # Load config
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/trainML.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    trainer = Trainer(cfg)

    # Run forward on val set, one batch just to test
    logits, paths = trainer.forward_all(ckpt_path=None, max_batches=1)


if __name__ == "__main__":
    main()
