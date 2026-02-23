import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

# --- Mamba wrapper (expects x: [B, N, D]) ---
try:
    from mamba_ssm import Mamba
except ImportError as e:
    Mamba = None  # keep import error until construction time

class MambaBlock(nn.Module):
    """
    Drop-in block using mamba-ssm. Input/Output: [B, N, D].
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, residual=False):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. `pip install mamba-ssm`")
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.residual = residual
    def forward(self, x):
        out = self.mamba(x)          # [B, N, D]
        return x + out if self.residual else out

class BasicSelfAttention(nn.Module):
    """
    Drop-in replacement for NystromAttention with standard MHA.
    Expects x: [B, N, D] and returns [B, N, D].
    """
    def __init__(self, dim, heads=8, dropout=0.5, residual=False):
        super().__init__()
        assert dim % heads == 0, "embed dim must be divisible by number of heads"
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.residual = residual

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x: [B, N, D]
        out, _ = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,  # shape [B, N] with True for PAD
            attn_mask=attn_mask,                # shape [N, N] or [B*H, N, N] if used
            need_weights=False
        )
        return x + out if self.residual else out


class TransLayer(nn.Module):

    def __init__(self,cfg, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.trans = str(cfg.get("trans", "att")).lower()

        if self.trans == "nystrom":
            self.attn = NystromAttention(
                dim=dim,
                dim_head=dim // 8,
                heads=8,
                num_landmarks=max(1, dim // 2),
                pinv_iterations=6,
                residual=True,
                dropout=0.1,
            )
        elif self.trans in {"att", "mha", "basic"}:
            self.attn = BasicSelfAttention(dim=dim, 
                heads=8, 
                dropout=0.3, 
                residual=False
            )

        elif self.trans in {"mamba", "ssm"}:
            # match your example: Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self.attn = MambaBlock(dim=dim, 
                d_state=64, 
                d_conv=5, 
                expand=3, 
                residual=True
            )
  
        else:
            raise ValueError(f"Unknown cfg['trans']={cfg['trans']!r}; use 'Nystrom' or 'att'")
                    
            
    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class AddapPPEG(nn.Module):
    def __init__(self, dim=512, mix_channels: bool = True):
        super().__init__()
        # depthwise 1x1 and 3x3
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=dim)
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        # optional channel mixing (pointwise 1x1, NOT depthwise)
        self.pw  = nn.Conv2d(dim, dim, kernel_size=1) if mix_channels else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.shape                     # x: [B, 1+H*W, C]
        assert H * W == (N - 1), "H*W must equal N-1"
        cls_token, feat_token = x[:, 0], x[:, 1:]
        feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)  # [B,C,H,W]

        y = feat + self.dw1(feat) + self.dw3(feat)  # local mixing + residual
        y = self.pw(y)                               # optional channel mixing

        y = y.flatten(2).transpose(1, 2)            # [B, H*W, C]
        out = torch.cat((cls_token.unsqueeze(1), y), dim=1)  # [B, 1+H*W, C]
        return out


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolMIL(nn.Module):
    """
    pool options:
      - "mean"         : [B,N,2048] -> [B,1,2048]
      - "max"          : [B,N,2048] -> [B,1,2048]
      - "attn"         : [B,N,2048] -> [B,1,2048] (learned attention)
      - "mean_median"  : [B,N,2048] -> [B,1,4096] (concat mean & median)
    """
    def __init__(self, cfg, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.emd_dim   = cfg.get("emd_dim", 512)
        self.dropout_p = cfg.get("dropout", 0.2)
        self.pool_type = cfg.get("pool", "mean")

        # infer pooled feature dim based on pool type
        if self.pool_type == "mean_median":
            pooled_in_dim = 2048 * 2
        else:
            pooled_in_dim = 2048

        # optional attention pooling
        if self.pool_type == "attn":
            self.attn = nn.Sequential(
                nn.Linear(2048, self.emd_dim),
                nn.Tanh(),
                nn.Linear(self.emd_dim, 1)
            )

        # two fully connected layers after pooling
        self.fc1 = nn.Sequential(
            nn.Linear(pooled_in_dim, self.emd_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        self.fc2 = nn.Linear(self.emd_dim, self.n_classes)

    def _pool(self, h):  # h: [B, N, 2048] -> [B, 1, D]
        if self.pool_type == "mean":
            out = h.mean(dim=1)                           # [B,2048]
            return out.unsqueeze(1)                       # [B,1,2048]
        elif self.pool_type == "max":
            out = h.max(dim=1).values                     # [B,2048]
            return out.unsqueeze(1)
        elif self.pool_type == "attn":
            a = self.attn(h).squeeze(-1)                  # [B,N]
            a = torch.softmax(a, dim=1)                   # [B,N]
            return torch.bmm(a.unsqueeze(1), h)           # [B,1,2048]
        elif self.pool_type == "mean_median":
            mu  = h.mean(dim=1)                           # [B,2048]
            med = h.median(dim=1).values                  # [B,2048]
            out = torch.cat([mu, med], dim=-1)            # [B,4096]
            return out.unsqueeze(1)                       # [B,1,4096]
        else:
            raise ValueError(f"Unknown pool='{self.pool_type}'")

    def forward(self, **kwargs):
        # Input: kwargs['data'] with shape [B, N, 2048]
        h = kwargs['data'].float()
        pooled = self._pool(h).squeeze(1)                 # [B, D]

        x = self.fc1(pooled)                              # [B, emd_dim]
        logits = self.fc2(x)                              # [B, n_classes]

        Y_prob = F.softmax(logits, dim=1)                 # use CrossEntropyLoss
        Y_hat  = torch.argmax(logits, dim=1)
        return {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}



if __name__ == "__main__":
    cfg = {
        "emd_dim": 1024,
        "input_dim": 2048,   # matches your data's last dim
        "input_morph_dim": 243,
        "cuda": 1,           # GPU id
        "ppeg": "norm",      # or "addaptive"
    }

    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")

    data = torch.randn((1, 16, 2048)).cuda()
    # data = torch.randn((1, 6000, 1024)).cuda()

    print("data shape-------------------", data.shape)
    model = PoolMIL(cfg, n_classes=2).cuda()
    # print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
