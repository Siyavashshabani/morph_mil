from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch, torch.nn as nn, torchvision as tv
from dataloader.dataloader import get_loaders 
# run_precompute.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from pathlib import Path
import shutil

from model.DINOv3Backbone import DinoV3Backbone
# -------------------------
# Frozen ResNet-101 encoder
# -------------------------
def build_frozen_backbone(cfg):
    if cfg["backbone"] == "resnet101":
        weights = tv.models.ResNet101_Weights.IMAGENET1K_V1
        m = tv.models.resnet101(weights=weights)
        m.fc = nn.Identity()  # 2048-d embedding
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m
    elif cfg["backbone"] == "DINOv3":
        # ---------------------- Config ----------------------
        dino_ckpt = "/home/sshabani/projects/segdino/web_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        repo_dir  = "/home/sshabani/projects/segdino/dinov3"
        backbone = DinoV3Backbone(
            repo_dir=repo_dir,
            arch="dinov3_vitl16",
            ckpt_path=dino_ckpt,
            pool="cls",          # "cls" for global CLS; "map" for spatial feature map
            freeze=True           # freeze for fixed feature extraction
        )    
        return backbone




@torch.no_grad()
def encode_and_save_item(encoder, item, out_dir: Path, device: str):
    x    = item["image"]                    # (16,3,256,256)
    path = item["path"]
    t_id = item["tertile_id"]
    t_str= item["tertile_str"]

    # ---- build key from file path ----
    p   = Path(str(path))
    key = f"{p.parent.name}_{p.stem}"      # or: key = p.stem
    # print("key------------------------------", key)
    # print("p--------------------------------", p)

    # exit()
    out_file = out_dir / f"{key}.pt"
    if out_file.exists():
        return

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.to(device, non_blocking=True).float()

    feats = encoder(x).float().cpu()       # (16,2048)
    print("feats.shape----------------------------", feats.shape)
    print("tertile_str----------------------------", t_str)
    
    # exit()
    torch.save({
        "id": key,
        "path": str(path),
        "tertile_id": int(t_id) if isinstance(t_id, (int, bool))
                     or (isinstance(t_id, torch.Tensor) and t_id.numel()==1) else t_id,
        "tertile_str": t_str,
        "feats": feats,
    }, out_file)


# -------------------------
# Iterate a (possibly batched) loader
# -------------------------
@torch.no_grad()
def precompute_for_loader(cfg, loader, out_dir, device, clean=True):
    out_dir = Path(out_dir)
    
    if clean and out_dir.exists():
        # --- safety guard: never nuke a root or home dir by mistake ---
        resolved = out_dir.resolve()
        if resolved == resolved.anchor or resolved == Path.home():
            raise ValueError(f"Refusing to remove dangerous path: {resolved}")
        shutil.rmtree(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ## define the model
    encoder = build_frozen_backbone(cfg).to(device).eval()

    for batch in loader:
        imgs = batch["image"]
        item = {
            "image": imgs,  # (16,3,256,256)
            "path": batch["path"][0] if isinstance(batch["path"], (list, tuple)) else batch["path"],
            "tertile_id": batch["tertile_id"][0] if isinstance(batch["tertile_id"], (list, tuple, torch.Tensor)) else batch["tertile_id"],
            "tertile_str": batch["tertile_str"][0] if isinstance(batch["tertile_str"], (list, tuple)) else batch["tertile_str"],
        }
        # print("batch['path'][0]-------------------",batch["path"][0])
        encode_and_save_item(encoder, item, out_dir, device)


            
# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Your CURRENT call, unchanged interface ===
    # You can keep cfg as-is; for speed & simplicity we override batch_size to 1 here.
    cfg = {
        "folder": "/home/sshabani/projects/balt_experiment/data/BAlt_Expirement",
        "excel":  "/home/sshabani/projects/balt_experiment/data/BAlt_Expirement/bAlt_scores_complete.xlsx",
        "id_column": "SampleID",
        "tertile_column": "Tertile",
        "pattern": "**/*DAPI.tif",
        "val_ratio": 0.2,
        "batch_size": 1,          # force 1 for clean per-item saves
        "num_workers": 4,
        "seed": 42,
        "backbone": "resnet101" #"DINOv3", #"resnet101", #
    }

    train_loader, val_loader = get_loaders(
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

    # Precompute features for both splits (in separate directories)
    precompute_for_loader(cfg, train_loader, out_dir=f"data/BAlt_Expirement/precompute{cfg['backbone']}/train", device=device)
    
    precompute_for_loader(cfg, val_loader,   out_dir=f"data/BAlt_Expirement/precompute{cfg['backbone']}/val",   device=device)

    print(f"✅ Done. Saved .pt feature files under features_{cfg['backbone']}_/train,val/")
