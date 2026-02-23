import torch

def cov_matrix(Z: torch.Tensor, unbiased: bool = True, eps: float = 0.0) -> torch.Tensor:
    N, D = Z.shape
    Zc = Z - Z.mean(dim=0, keepdim=True)
    denom = (N - 1) if unbiased else N
    C = (Zc.T @ Zc) / denom
    if eps > 0:
        C = C + eps * torch.eye(D, device=Z.device, dtype=Z.dtype)
    return C

def cov_fro_loss(z: torch.Tensor, z_prime: torch.Tensor, unbiased: bool = True,
                 squared: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    Returns a scalar loss based on ||Cov(z) - Cov(z')||_F (or squared).
    """
    if z.shape != z_prime.shape:
        raise ValueError(f"Shape mismatch: {z.shape} vs {z_prime.shape}")

    Cz = cov_matrix(z, unbiased=unbiased)
    Cz2 = cov_matrix(z_prime, unbiased=unbiased)
    D = Cz.shape[0]

    diff = Cz - Cz2

    if squared:
        loss = (diff * diff).sum()          # == ||diff||_F^2
        if normalize:
            loss = loss / (D * D)           # optional scale-invariant
    else:
        loss = torch.linalg.norm(diff, ord="fro")  # == ||diff||_F
        if normalize:
            loss = loss / D                 # optional scale-invariant

    return loss

# Example
device = "cuda" if torch.cuda.is_available() else "cpu"
z = torch.randn(256, 1024, device=device)
z_prime = torch.randn(256, 1024, device=device)

loss = cov_fro_loss(z, z_prime, squared=False, normalize=True)
print("loss:", loss.item())













import torch
import torch.nn as nn
import torch.nn.functional as F

def cov_matrix(Z: torch.Tensor, unbiased: bool = True, eps: float = 0.0) -> torch.Tensor:
    """
    Z: [N, D] -> Cov(Z): [D, D]
    """
    if Z.dim() != 2:
        raise ValueError(f"Expected [N, D], got {tuple(Z.shape)}")

    N, D = Z.shape
    if unbiased and N < 2:
        raise ValueError("Need N >= 2 for unbiased covariance (N-1 in denominator).")

    Zc = Z - Z.mean(dim=0, keepdim=True)
    denom = (N - 1) if unbiased else N
    C = (Zc.T @ Zc) / denom  # [D, D]

    if eps > 0:
        C = C + eps * torch.eye(D, device=Z.device, dtype=Z.dtype)

    return C

def cov_fro_loss_from_two_views(z2nd: torch.Tensor,
                               unbiased: bool = True,
                               squared: bool = True,
                               normalize: bool = True,
                               eps: float = 0.0) -> torch.Tensor:
    """
    z2nd: [2, N, D] (two views)
    returns scalar covariance loss
    """
    
    if z2nd.dim() != 3 or z2nd.size(0) != 2:
        raise ValueError(f"Expected z shape [2, N, D], got {tuple(z2nd.shape)}")

    # compute in float32 for numerical stability (esp. if autocast fp16 is on)
    z0 = z2nd[0].float()  # [N, D]
    z1 = z2nd[1].float()

    C0 = cov_matrix(z0, unbiased=unbiased, eps=eps)
    C1 = cov_matrix(z1, unbiased=unbiased, eps=eps)

    diff = C0 - C1
    D = diff.shape[0]

    if squared:
        loss = (diff * diff).sum()  # ||diff||_F^2
        if normalize:
            loss = loss / (D * D)
    else:
        loss = torch.linalg.norm(diff, ord="fro")  # ||diff||_F
        if normalize:
            loss = loss / D

    return loss


class CEPlusCovLoss(nn.Module):
    def __init__(self, lambda_cov: float = 1.0,
                 unbiased: bool = True,
                 squared: bool = True,
                 normalize: bool = True,
                 eps: float = 0.0):
        super().__init__()
        self.lambda_cov = float(lambda_cov)
        self.unbiased = unbiased
        self.squared = squared
        self.normalize = normalize
        self.eps = eps
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_prob: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        logits: ideally [B, C], but we will handle [B, C, ...] or [K, C] cases
        y:      [B] (class indices)
        z:      [2, N, 1024] (two views)
        """

        # ---- 1) Make y correct dtype/shape for CE ----
        # If y comes as [B,1] -> [B]
        if y.dim() > 1:
            y = y.view(-1)
        y = y.long()


        y_prob_clean = y_prob[0].unsqueeze(0)
        y= y.view(-1).long()


        ce = F.nll_loss(y_prob_clean, y) 
        # print("ce-----------------------------------", ce)

        # ---- 3) Ensure z is exactly 2 views ----
        # if z.size(0) > 2:
        #     z = z[:2]  # from [3, N, D] -> [2, N, D]

        # cov = cov_fro_loss_from_two_views(
        #     z, unbiased=self.unbiased, squared=self.squared,
        #     normalize=self.normalize, eps=self.eps
        # )

        total = ce #+ self.lambda_cov * cov
        return total, {"ce": ce.detach()} #, "cov": cov.detach()
###################################################################################################
################################################################################################### 

