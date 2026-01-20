import torch
import torch.nn as nn
import torch.nn.functional as F

# --- paste your SupConLoss here (unchanged) ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # print("features------------------------", features.shape, features[:, 0].shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def grad_ok(t: torch.Tensor) -> bool:
    return (t.grad is not None) and torch.isfinite(t.grad).all().item()

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bsz, n_views, dim = 8, 1000, 128
    loss_fn_all = SupConLoss(temperature=0.1, contrast_mode="all").to(device)
    loss_fn_one = SupConLoss(temperature=0.1, contrast_mode="one").to(device)

    # IMPORTANT: keep a leaf tensor x so x.grad is populated
    x = torch.randn(bsz, n_views, dim, device=device, requires_grad=True)
    features = F.normalize(x, dim=-1)  # features is non-leaf, x is leaf

    print(f"Device: {device}")
    print(f"features shape: {features.shape}")

    # 1) Unsupervised
    loss_unsup = loss_fn_all(features)
    print(f"[Unsupervised] loss: {loss_unsup.item():.6f}")
    loss_unsup.backward(retain_graph=True)
    print(f"[Unsupervised] grad finite on x? {grad_ok(x)}")
    x.grad.zero_()

    # 2) Supervised labels
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    loss_sup_all = loss_fn_all(features, labels=labels)
    loss_sup_one = loss_fn_one(features, labels=labels)
    print(f"[Supervised/all] loss: {loss_sup_all.item():.6f}")
    print(f"[Supervised/one] loss: {loss_sup_one.item():.6f}")
    loss_sup_all.backward(retain_graph=True)
    print(f"[Supervised] grad finite on x? {grad_ok(x)}")
    x.grad.zero_()

    # 3) Custom mask pairs
    mask = torch.zeros((bsz, bsz), device=device)
    pairs = [(0,1), (2,3), (4,5), (6,7)]
    for i, j in pairs:
        mask[i, j] = 1.0
        mask[j, i] = 1.0

    loss_mask = loss_fn_all(features, mask=mask)
    print(f"[Custom mask] loss: {loss_mask.item():.6f}")
    loss_mask.backward(retain_graph=True)
    print(f"[Custom mask] grad finite on x? {grad_ok(x)}")
    x.grad.zero_()

    # 4) Edge case: singleton positives (should not NaN)
    x2 = torch.randn(4, 1, dim, device=device, requires_grad=True)
    f2 = F.normalize(x2, dim=-1)
    labels_edge = torch.tensor([0, 1, 1, 2], device=device)
    loss_edge = loss_fn_all(f2, labels=labels_edge)
    print(f"[Edge case singleton positives] loss: {loss_edge.item():.6f}")
    loss_edge.backward()
    print(f"[Edge case] grad finite on x2? {grad_ok(x2)}")

if __name__ == "__main__":
    main()
