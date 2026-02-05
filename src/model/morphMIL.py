import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from .adaptorMorph import AdaptorViT
from .adaptorFNOMorph import AdaptorFNO

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


class MorphMIL(nn.Module):
    def __init__(self, cfg, n_classes):
        super(MorphMIL, self).__init__()
        self.emd_dim = cfg.get("emd_dim")
        self.input_dim = cfg.get("input_dim" )
        self.game_theory = cfg.get("game_theory", True )

        ## what branch: 
        self.model_branch = cfg.get("model_branch" )

        ## define the gpu ids
        gpu_id = cfg.get("cuda", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")


        if cfg["ppeg"]=="norm":
            self.pos_layer = PPEG(dim=self.emd_dim)
        elif cfg["ppeg"]=="addaptive":
            self.pos_layer = AddapPPEG(dim=self.emd_dim, mix_channels=False )
            
        self._fc1 = nn.Sequential(nn.Linear(self.input_dim, self.emd_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emd_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(cfg, dim=self.emd_dim )
        # self.layer11 = TransLayer(cfg, dim=self.emd_dim )

        self.layer2 = TransLayer(cfg, dim=self.emd_dim )
        # self.layer22 = TransLayer(cfg, dim=self.emd_dim )

        self.norm = nn.LayerNorm(self.emd_dim)
        # self._fc2 = nn.Linear(1024, 512)
        self._fc3 = nn.Linear(self.emd_dim, self.n_classes)
        self.dropout = nn.Dropout(p=0.2)          # try 0.2–0.5

        ## 
        self.cancat_type = cfg.get("cancat_type", "simple")
        ############################### AdaptorMorph is here
        if self.model_branch=="morph" or self.model_branch=="both":
            if cfg.get("adaptor")=="vit":
                self.adaptor_morph = AdaptorViT(cfg)
            elif cfg.get("adaptor")=="fno": 
                self.adaptor_morph = AdaptorFNO(cfg)

    def forward(self, data, h_morph):

        ## define the inputs        
        h = data.float() #[B, n, 1024]
        
        # print("output of h_morph ---------------------", h_morph.shape)
        # exit()

        if self.model_branch=="morph":
            h_morph = h_morph.float()        
            # print("input of h ----------------------------", h.shape)
            h_morph = self.adaptor_morph(h_morph)
            h = h_morph 

        elif self.model_branch=="image":      
            ################################# base MIL
            # print("h--------------------------------------------", h.shape)
            h = self._fc1(h) #[B, n, 512]
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
            
            #---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)
            h = torch.cat((cls_tokens, h), dim=1)

            #---->Translayer x1
            h = self.layer1(h) #[B, N, 512]
            
            #---->PPEG
            h = self.pos_layer(h, _H, _W) #[B, N, 512]
            
            #---->Translayer x2
            h = self.layer2(h) #[B, N, 512]

        
        elif self.model_branch=="both":   
            h_morph = h_morph.float()        
            h_morph = self.adaptor_morph(h_morph)
            
            ################################# How to concate the feats and morph
            if self.cancat_type=="simple":
                if self.game_theory:
                    Z_M = torch.cat((h[0,:,:],h_morph[0,:,:]),dim=1).unsqueeze(0) 
                    Zp_M = torch.cat((h[1,:,:],h_morph[0,:,:]),dim=1).unsqueeze(0) 
                    Z_Mp = torch.cat((h[0,:,:],h_morph[1,:,:]),dim=1).unsqueeze(0) 
                    h = torch.cat((Z_M, Zp_M, Z_Mp),dim=0 )    
                else: 
                    h = torch.cat((h,h_morph),dim=2)
                    
            elif self.cancat_type=="one_to_all":
                h_morph = h_morph.repeat(h.size(0), 1, 1)            # [3, 7875, 246]
                h = torch.cat([h, h_morph], dim=-1)            
            # print("output of concate ---------------------", h.shape)
            # exit()
            
            ################################# base MIL
            h = self._fc1(h) #[B, n, 512]
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
            
            #---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)
            h = torch.cat((cls_tokens, h), dim=1)

            #---->Translayer x1
            h = self.layer1(h) #[B, N, 512]
            
            #---->PPEG
            h = self.pos_layer(h, _H, _W) #[B, N, 512]
            
            #---->Translayer x2
            h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        logits = self._fc3(h)              # [B, n_classes]
        # print("in model logits---------", logits.shape)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # print("Y_prob----------------------", Y_prob)
        return results_dict

if __name__ == "__main__":
    cfg = {
        "emd_dim": 1024,
        "input_dim": 2048,   # matches your data's last dim
        "input_morph_dim": 243,
        "cuda": 1,           # GPU id
        "ppeg": "norm",      # or "addaptive"
    }

    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")

    input_ = {}
    input_["data"] = torch.randn((1, 16, 1024), device=device)
    input_["morph"] = torch.randn((1, 16, 243), device=device)

    model = MorphMIL(cfg=cfg, n_classes=2).to(device)

    results_dict = model(data=input_["data"], morph=input_["morph"])
    print(results_dict)
