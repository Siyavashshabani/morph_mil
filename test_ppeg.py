import torch
import torch.nn as nn

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)  # depthwise
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)  # depthwise
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)  # depthwise

    def forward(self, x, H, W):
        B, _, C = x.shape                       # x: [B, N, C]
        cls_token, feat_token = x[:, 0], x[:, 1:]   # [B, C], [B, N-1, C]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)  # -> [B, C, H, W]
        print("cnn_feat---------------------", cnn_feat.shape)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        print("after projection----------------------------", x.shape)
        x = x.flatten(2).transpose(1, 2)        # -> [B, H*W, C]
        print("after flattening----------------------------", x.shape)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)  # -> [B, 1+H*W, C]
        return x

def test_ppeg():
    B, N, C = 1, 17, 512     # input shape [1, 17, 512]
    H = W = 4                # because N-1 = 16 = 4*4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make dummy input
    x = torch.randn(B, N, C, device=device, requires_grad=True)

    # Build module
    ppeg = PPEG(dim=C).to(device)
    ppeg.train()   # test that autograd works too

    # Sanity: H*W must equal N-1
    assert H * W == (N - 1), f"H*W must equal N-1; got {H}*{W} != {N-1}"

    # Forward
    y = ppeg(x, H, W)
    print("input shape :", x.shape)
    print("output shape:", y.shape)

    # Check shape matches input (should be [1, 17, 512])
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    # Quick grad check
    loss = y.pow(2).mean()
    loss.backward()
    # verify some params received gradients
    grads_ok = all(p is None or p.grad is not None for p in [ppeg.proj.weight, ppeg.proj1.weight, ppeg.proj2.weight])
    print("gradients ok:", grads_ok)

if __name__ == "__main__":
    test_ppeg()
