import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url).convert("RGB")

ckpt = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32  # <- simplest: keep model & inputs in fp32

processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt, dtype=dtype).to(device).eval()

inputs = processor(images=image, return_tensors="pt")
# make sure every tensor input is on same device & dtype
inputs = {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)

print("Available output keys:", list(outputs.keys()))
if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
    print("pooler_output:", tuple(outputs.pooler_output.shape))
if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
    print("last_hidden_state:", tuple(outputs.last_hidden_state.shape))
