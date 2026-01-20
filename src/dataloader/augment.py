# --- augs.py --------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os, time
from pathlib import Path

def _as_btd(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.dim() == 2:         # [T,D]
        return x.unsqueeze(0), True
    if x.dim() == 3:         # [B,T,D]
        return x, False
    raise ValueError(f"Expected [T,D] or [B,T,D], got {tuple(x.shape)}")

class TensorAugment(nn.Module):
    """
    Feature-space augs for [T,D] / [B,T,D] bags (e.g., T=16, D=2048).
    Enable/disable by probability; keeps shape unchanged.
    """
    def __init__(
        self,
        # token-level
        p_token_drop: float = 0.2, token_drop_frac: float = 0.15, token_drop_mode: str = "zero",
        p_token_shuffle: float = 0.2, local_shuffle: bool = True, local_window: int = 4,
        p_token_span_mask: float = 0.2, token_span_frac: float = 0.25,

        # token-mix (token-level mixup)
        p_token_mix: float = 0.1, token_mix_frac: float = 0.05, token_mix_alpha: float = 0.05,

        # feature-level
        p_feat_jitter: float = 0.5, feat_jitter_sigma: float = 0.02,
        p_feat_band_mask: float = 0.3, feat_band_frac: float = 0.1,
        p_feat_affine: float = 0.3, affine_scale_std: float = 0.1, affine_shift_std: float = 0.02,
        # rectangular cutout in (T x D)
        p_rect_cutout: float = 0.2, rect_token_frac: float = 0.25, rect_feat_frac: float = 0.15,
    ):
        super().__init__()
        self.p_token_drop, self.token_drop_frac, self.token_drop_mode = p_token_drop, token_drop_frac, token_drop_mode
        self.p_token_shuffle, self.local_shuffle, self.local_window = p_token_shuffle, local_shuffle, local_window
        self.p_token_span_mask, self.token_span_frac = p_token_span_mask, token_span_frac

        self.p_token_mix, self.token_mix_frac, self.token_mix_alpha = p_token_mix, token_mix_frac, token_mix_alpha

        self.p_feat_jitter, self.feat_jitter_sigma = p_feat_jitter, feat_jitter_sigma
        self.p_feat_band_mask, self.feat_band_frac = p_feat_band_mask, feat_band_frac
        self.p_feat_affine, self.affine_scale_std, self.affine_shift_std = p_feat_affine, affine_scale_std, affine_shift_std
        self.p_rect_cutout, self.rect_token_frac, self.rect_feat_frac = p_rect_cutout, rect_token_frac, rect_feat_frac

    def _token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """
        TokenMix: replace a subset of tokens with convex mixtures of other tokens (within the same bag).
        x: [B, T, D]
        """
        B, T, D = x.shape
        dev = x.device

        k = max(1, int(T * self.token_mix_frac))

        beta = torch.distributions.Beta(self.token_mix_alpha, self.token_mix_alpha)
        lam = beta.sample((B, 1, 1)).to(dev)  # [B,1,1]

        for b in range(B):
            idx = torch.randperm(T, device=dev)[:k]
            jdx = torch.randperm(T, device=dev)[:k]
            x[b, idx] = lam[b] * x[b, idx] + (1.0 - lam[b]) * x[b, jdx]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, added_batch = _as_btd(x)
        B, T, D = x.shape
        dev = x.device

        # TokenShuffle
        if torch.rand((), device=dev) < self.p_token_shuffle:
            if self.local_shuffle and self.local_window > 1:
                for b in range(B):
                    for s in range(0, T, self.local_window):
                        e = min(T, s + self.local_window)
                        x[b, s:e] = x[b, s:e][torch.randperm(e - s, device=dev)]
            else:
                perm = torch.randperm(T, device=dev)
                x = x[:, perm]

        # TokenMix (do after shuffle)
        if torch.rand((), device=dev) < self.p_token_mix:
            x = self._token_mix(x)

        # Feature jitter
        if torch.rand((), device=dev) < self.p_feat_jitter:
            x = x + torch.randn_like(x) * self.feat_jitter_sigma

        # Feature affine (MixStyle-lite)
        if torch.rand((), device=dev) < self.p_feat_affine:
            gamma = 1.0 + torch.randn(B, 1, D, device=dev) * self.affine_scale_std
            beta  = torch.randn(B, 1, D, device=dev) * self.affine_shift_std
            x = gamma * x + beta

        return x.squeeze(0) if added_batch else x

