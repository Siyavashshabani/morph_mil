import torch, torchvision, torchaudio, PIL, numpy, sys, platform


# Robust way to get the installed package version for a module
from importlib.metadata import version, packages_distributions, PackageNotFoundError
import importlib

module_name = "nystrom_attention"
mod = importlib.import_module(module_name)
print("module file:", getattr(mod, "__file__", "<none>"))

dist_names = packages_distributions().get(module_name, [])
if dist_names:
    for d in dist_names:
        try:
            print(f"{d} version:", version(d))
        except PackageNotFoundError:
            print(f"{d} version: <not found>")
else:
    # Fallback: try common naming variants and a quick scan
    candidates = {
        module_name, module_name.replace("_", "-"), module_name.replace("-", "_")
    }
    printed = False
    for d in candidates:
        try:
            print(f"{d} version:", version(d)); printed = True
        except PackageNotFoundError:
            pass
    if not printed:
        print("Could not map module to a distribution (editable/local install?).")




print("python:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
print("torchvision:", getattr(torchvision, "__version__", None))
print("torchaudio:", getattr(torchaudio, "__version__", None))
print("PIL:", PIL.__version__)
print("numpy:", numpy.__version__)
try:
    import timm; print("timm:", timm.__version__)
except Exception:
    print("timm: N/A")
try:
    import sklearn; print("sklearn:", sklearn.__version__)
except Exception:
    print("sklearn: N/A")
