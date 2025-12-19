import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def _as_activation(act):
    if isinstance(act, str):
        act = act.lower()
        return {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}.get(act, nn.ReLU())
    if isinstance(act, nn.Module):
        return act
    raise TypeError("act must be a string or an nn.Module")

class GATE_AB_MIL(nn.Module):
    def __init__(self, L=1024, D=128, num_classes=2, act="relu", dropout=0.0, bias=False, in_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = float(dropout)
        self.in_dim = in_dim
        self.L = L
        self.D = D 
        self.K = 1

        act_mod = _as_activation(act)

        # instance encoder to L
        feat_layers = [nn.Linear(in_dim, L), act_mod]
        if self.dropout > 0:
            feat_layers.append(nn.Dropout(self.dropout))
        self.feature = nn.Sequential(*feat_layers)

        # gated attention
        attn_a = [nn.Linear(L, D, bias=bias), act_mod]
        attn_b = [nn.Linear(L, D, bias=bias), nn.Sigmoid()]
        if self.dropout > 0:
            attn_a.append(nn.Dropout(self.dropout))
            attn_b.append(nn.Dropout(self.dropout))
        self.attention_a = nn.Sequential(*attn_a)
        self.attention_b = nn.Sequential(*attn_b)
        self.attention_c = nn.Linear(D, self.K, bias=bias)

        # classifier (since K=1, use L -> C)
        self.classifier = nn.Linear(L, num_classes, bias=bias)

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=True, return_WSI_feature=True):
        """
        x: [B, N, in_dim]  (e.g., [B, 16, 2048])
        """
        forward_return = {}
        B, N, _ = x.shape

        x = self.feature(x)                 # [B, N, L]
        a = self.attention_a(x)             # [B, N, D]
        b = self.attention_b(x)             # [B, N, D]
        # print("a, b----------------------", a.shape, b.shape)
        A = self.attention_c(a * b)         # [B, N, K]
        A_ori = A.clone()                   # pre-softmax

        A = A.transpose(-1, -2)             # [B, K, N]
        # print("A------------------------", A.shape )
        A = F.softmax(A, dim=-1)            # softmax over instances
        # print("A, x---------------------", A.shape, x.shape )
        H = torch.matmul(A, x)              # [B, K, L]
        # print("H------------------------", H.shape)
        # exit()

        H = H.squeeze(1)                    # [B, L] since K=1

        logits = self.classifier(H)         # [B, C]
        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = H.unsqueeze(1)  # [B,1,L] for consistency
        if return_WSI_attn:
            forward_return["WSI_attn"] = A_ori              # [B, N, 1]
        return forward_return



def quick_test(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(123)

    # config consistent with your input [B,16,2048]
    B = 4
    num_classes = 2
    model = GATE_AB_MIL(
        L=512, D=128, num_classes=num_classes,
        act='relu',         # <-- use string so attention_a gets activation
        dropout=0.1, bias=False,
        in_dim=2048         # <-- must match x.shape[-1]
    ).to(device)

    # fake batch: [B, 16, 2048]
    x = torch.randn(B, 16, 2048, device=device)

    # forward (also request returns for inspection)
    out = model(x, return_WSI_attn=True, return_WSI_feature=True)

    logits = out["logits"]          # [B, 1, C] or [1, C] if B==1 and your squeeze(0) triggers
    WSI_feat = out["WSI_feature"]   # [B, 1, 512]
    WSI_attn = out["WSI_attn"]      # [B, 16, 1] (pre-softmax)

    print("logits shape:", tuple(logits.shape))
    print("WSI_feature shape:", tuple(WSI_feat.shape))
    print("WSI_attn shape:", tuple(WSI_attn.shape))

    # make labels and compute a loss; squeeze the singleton "K" dim
    y = torch.randint(0, num_classes, (B,), device=device)
    loss = F.cross_entropy(logits.squeeze(1), y)
    print("loss:", float(loss.item()))

    # backward to verify gradients flow
    loss.backward()
    # quick gradient check on a few parameters
    grad_ok = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_ok.append((name, p.grad.norm().item()))
            if len(grad_ok) >= 3:
                break
    print("sample grad norms:", grad_ok)

if __name__ == "__main__":
    quick_test()
