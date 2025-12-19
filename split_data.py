#!/usr/bin/env python3
import re, shutil, random
from pathlib import Path

# ==== CONFIG ====
ROOT = Path(r"/home/sshabani/projects/balt_experiment/data/BAlt_Expirement/precomputeResNet/train_1")  # <-- change to your folder (works on Linux/macOS too)
OUT  = ROOT                              # put train/ and val/ under ROOT
VAL_RATIO = 0.2                          # 20% validation
SEED = 42
DRY_RUN = False                           # set False to actually move files
GLOB_PATTERN = "*.pt"                    # match .pt files
# ===============

random.seed(SEED)

# Extract mouse id (e.g., p275) from filenames like:
# Week1_A1819-p275R-01_DAPI_1.pt  -> "p275"
PID_REGEX = re.compile(r"p(\d+)", re.IGNORECASE)

files = sorted(ROOT.glob(GLOB_PATTERN))
if not files:
    raise SystemExit(f"No files matching {GLOB_PATTERN} in {ROOT}")

def get_pid(path: Path):
    m = PID_REGEX.search(path.name)
    return f"p{m.group(1)}" if m else None

# Build per-file group labels (mouse ids)
file_pid = []
skipped = []
for f in files:
    pid = get_pid(f)
    if pid is None:
        skipped.append(f)
    else:
        file_pid.append((f, pid))

if skipped:
    print(f"WARNING: {len(skipped)} files have no pID and will be ignored:")
    for s in skipped[:10]:
        print("  -", s.name)
    if len(skipped) > 10:
        print("  ...")

# Group files by pid
from collections import defaultdict
by_pid = defaultdict(list)
for f, pid in file_pid:
    by_pid[pid].append(f)

pids = sorted(by_pid.keys())
print(f"Found {len(file_pid)} usable files across {len(pids)} mouse IDs.")

# Split by pid so no leakage
num_val_pids = max(1, int(round(len(pids) * VAL_RATIO)))
val_pids = set(random.sample(pids, num_val_pids))
train_pids = [p for p in pids if p not in val_pids]

# Collect files for each split
train_files = [f for p in train_pids for f in by_pid[p]]
val_files   = [f for p in val_pids   for f in by_pid[p]]

print("\n=== SPLIT SUMMARY ===")
print(f"Train: {len(train_files)} files, {len(train_pids)} pIDs")
print(f"Val  : {len(val_files)} files, {len(val_pids)} pIDs")

# Sanity check: no overlap of pids
assert set(train_pids).isdisjoint(val_pids), "Leakage: a pid is in both splits!"

# Prepare output directories
train_dir = OUT / "train"
val_dir   = OUT / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Move (or preview) files
def move_files(file_list, dest_dir):
    moved = 0
    for src in file_list:
        dst = dest_dir / src.name
        if DRY_RUN:
            print(f"[DRY] {src.name}  ->  {dest_dir.name}/")
        else:
            shutil.move(str(src), str(dst))
            moved += 1
    return moved

print("\nMoving files...")
moved_train = move_files(train_files, train_dir)
moved_val   = move_files(val_files,   val_dir)

if DRY_RUN:
    print("\nDRY RUN complete. Set DRY_RUN=False to actually move files.")
else:
    print(f"\nDone. Moved {moved_train} train and {moved_val} val files.")
