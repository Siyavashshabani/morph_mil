import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from .augment import TensorAugment

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


class AdaptorViT(nn.Module):
    def __init__(self, cfg ):
        super(AdaptorViT, self).__init__()
        self.emd_dim = cfg.get("emd_morph_dim", 512)
        self.input_dim = cfg.get("input_morph_dim", 1024)
        self.aug_flag = cfg.get("aug_morph", True)
        ## define the gpu ids
        gpu_id = cfg.get("cuda", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")

        self._fc1 = nn.Sequential(nn.Linear(self.input_dim, self.emd_dim), nn.ReLU())
        self.layer1 = TransLayer(cfg, dim=self.emd_dim )
        # self.layer2 = TransLayer(cfg, dim=self.emd_dim )

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

    def forward(self, data):
        
        h = data.float() #[B, n, 1024]

        # print("h.shape----------------------", h.shape)
        h = self._fc1(h) #[B, n, 512]

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        # h = self.layer2(h) #[B, N, 512]

        if self.aug_flag == True: 
            # print("h.dtype------------------", h.dtype)
            h_aug1 = self.aug(h.clone())
            h_aug2 = self.aug_light(h.clone())
            h = torch.stack([h, h_aug1, h_aug2], dim=0).squeeze(1)
            # print("aug flag---------------------------------------------------")         
            # print("h----------------------------------------", h.shape)
        return h 
    


######################################################################### 
######################################################################### Fourier Neural Operator(FNO)




    
if __name__ == "__main__":
    cfg = {
        "emd_dim": 1024,
        "input_morph_dim": 243,   # matches your data's last dim
        "cuda": 1,           # GPU id
        "ppeg": "norm",      # or "addaptive"
    }

    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")

    data = torch.randn((1, 16, 243), device=device)

    model = AdaptorViT(cfg=cfg).to(device)

    results_dict = model(data=data)
    print(results_dict.shape)
