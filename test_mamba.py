import torch, triton
# from model.mamba import MambaBlock, has_mamba
from mamba_ssm import Mamba

print("torch:", torch.__version__)
print("triton ok:", triton.__version__)
x = torch.randn(2, 64, 16).cuda() if torch.cuda.is_available() else torch.randn(2,64,16)
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2)
m = m.cuda() if torch.cuda.is_available() else m
y = m(x)
print("y.shape:", y.shape)




# source ~/.venvs/balt_mamba/bin/activate
# which python   # should point to ~/.venvs/balt_mamba/bin/python

