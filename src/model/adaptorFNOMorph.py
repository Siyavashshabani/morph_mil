
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fno.models.fno import FNO1d
from augment import TensorAugment

class AdaptorFNO(nn.Module):
    def __init__(self, cfg ):
        super(AdaptorFNO, self).__init__()
        self.emd_dim = cfg.get("emd_morph_dim", 512)
        self.input_dim = cfg.get("input_morph_dim", 1024)
        self.aug_flag = cfg.get("aug_morph", True)
        ## define the gpu ids
        gpu_id = cfg.get("cuda", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        self.fno = FNO1d(
            n_modes_height=self.emd_dim,
            hidden_channels=self.emd_dim,
            in_channels=self.input_dim ,      # <-- 1024 channels
            out_channels=1,     # choose what you want to predict per point
            n_layers=4,
            domain_padding=None,
        )
        
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
        # Convert [B, N, F] -> [B, F, N] for FNO1d
        h = h.permute(0, 2, 1).contiguous()

        h = self.fno(h)   # expected: [B, out_channels, N]

        # Convert back to tokens-first if you want: [B, N, out_channels]
        h = h.permute(0, 2, 1).contiguous()
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
        "emd_morph_dim": 512,
        "input_morph_dim": 243,   # matches your data's last dim
        "cuda": 1,           # GPU id
        "ppeg": "norm",      # or "addaptive"
    }

    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() else "cpu")

    data = torch.randn((1, 10000, 243), device=device)

    model = AdaptorFNO(cfg=cfg).to(device)

    results_dict = model(data=data)
    print(results_dict.shape)
