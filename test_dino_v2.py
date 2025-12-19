# dino_v3_single_url_binary.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

ckpt = "facebook/dinov3-vitb16-pretrain-lvd1689m"
url  = "http://images.cocodataset.org/val2017/000000039769.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = (torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported()
          else (torch.float16 if device=="cuda" else torch.float32))

# 1) Model + processor (freeze backbone)
proc = AutoImageProcessor.from_pretrained(ckpt, token=True)
backbone = AutoModel.from_pretrained(ckpt, token=True, dtype=dtype).to(device).eval()
for p in backbone.parameters(): p.requires_grad = False
head = nn.Linear(backbone.config.hidden_size, 2).to(device)  # 2 classes (0/1)

# 2) Make tiny “dataset” from the single URL image (label = 1)
img = load_image(url).convert("RGB")
pv  = proc(images=img, return_tensors="pt")["pixel_values"]          # [1,3,H,W]
xtr = pv.repeat(8, 1, 1, 1)   # 8 train copies
ytr = torch.ones(8, dtype=torch.long)  # label=1
xva = pv.repeat(4, 1, 1, 1)   # 4 val copies
yva = torch.ones(4, dtype=torch.long)

dl_tr = DataLoader(TensorDataset(xtr, ytr), batch_size=4, shuffle=True)
dl_va = DataLoader(TensorDataset(xva, yva), batch_size=4, shuffle=False)

opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
ce  = nn.CrossEntropyLoss()

def run_epoch(dl, train=True):
    head.train(train)
    tot, correct, n = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)

        # 1) Backbone: no grad (NOT inference_mode)
        with torch.no_grad():
            feats = backbone(pixel_values=xb).pooler_output  # [B,768]
            print("feats---------------------------", feats.shape)
        # 2) Head: normal autograd
        feats = feats.float()                 # make sure dtype matches head
        logits = head(feats)                  # [B,2]
        loss = ce(logits, yb)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        tot += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return tot/n, correct/n


for epoch in range(3):
    tr_loss, tr_acc = run_epoch(dl_tr, True)
    va_loss, va_acc = run_epoch(dl_va, False)
    print(f"epoch {epoch}: train acc={tr_acc:.3f}  val acc={va_acc:.3f}")