import torch
from transformers import AutoImageProcessor, AutoModel

ckpt = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# load as you already do
processor = AutoImageProcessor.from_pretrained(ckpt, token=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
          else (torch.float16 if device == "cuda" else torch.float32))

model = AutoModel.from_pretrained(ckpt, token=True, dtype=dtype).to(device).eval()

# --- save as .pth (CPU, float32 for portability) ---
save_path = "/home/sshabani/projects/segdino/web_pth/dinov3-vitb16-pretrain-lvd1689m"
cpu_fp32_state = {k: v.detach().to("cpu", dtype=torch.float32) for k, v in model.state_dict().items()}
torch.save(cpu_fp32_state, save_path)
print(f"Saved weights to {save_path}")