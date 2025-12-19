import torch
import torch.nn as nn

# Support both exports: some tags expose Mamba, others Mamba2
_MambaImpl = None
_IMPORT_ERR = None
try:
    from mamba_ssm import Mamba as _MambaImpl
except Exception as e1:
    try:
        from mamba_ssm import Mamba2 as _MambaImpl
    except Exception as e2:
        _IMPORT_ERR = (e1, e2)

def has_mamba() -> bool:
    return _MambaImpl is not None

class MambaBlock(nn.Module):
    """
    Thin adapter so the rest of your repo never imports upstream symbols directly.
    Input:  x [B, L, d_model]  ->  Output: [B, L, d_model]
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _MambaImpl is None:
            raise ImportError(
                "mamba-ssm not available in the current env.\n"
                "Activate your env and install:\n"
                "  source ~/.venvs/balt_mamba/bin/activate && pip install -e third_party/mamba-ssm\n"
                f"Original import errors: {_IMPORT_ERR}"
            )
        self.core = _MambaImpl(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect [B, L, d_model]
        return self.core(x)
