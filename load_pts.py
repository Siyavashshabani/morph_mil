from pathlib import Path
import torch

folder = Path("data/BAlt_Expirement/precomputeresnet101/train")  # <- change me

for f in sorted(folder.glob("*.pt")):
    obj = torch.load(f, map_location="cpu")  # safe on any machine
    tstr = obj.get("tertile_str", None)
    tid  = obj.get("tertile_id", None)
    kid  = obj.get("id", None)
    pth  = obj.get("path", str(f))
    print(f"{f.name} | id={kid} | tertile_id={tid} | tertile_str={tstr} | path={pth}")
