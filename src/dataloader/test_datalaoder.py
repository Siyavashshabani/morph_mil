#!/usr/bin/env python3
"""
Read a TIFF image, display it, and save as PNG in an output directory.
"""

import os
import tifffile
import matplotlib.pyplot as plt

def main():
    # Input TIFF file
    file_path = "/home/sshabani/projects/balt_experiment/data/BAlt_Expirement/Week1/A1821-p694-02_DAPI.tif"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read image
    image = tifffile.imread(file_path)

    # Print info
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image min and max:", image.min(), image.max())

    # Output directory
    out_dir = "./output/whole_sample"
    os.makedirs(out_dir, exist_ok=True)

    # Derive PNG filename
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(out_dir, f"{base}.png")

    # Plot and save
    plt.figure(figsize=(6, 6))
    if image.ndim > 2:  # e.g., z-stack, show first slice
        plt.imshow(image[0], cmap="gray")
        plt.title(f"{base} [slice 0]")
    else:
        plt.imshow(image, cmap="gray")
        plt.title(base)
    plt.axis("off")
    plt.tight_layout()

    # Save to PNG
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved image to {out_path}")

    # Optionally also show
    plt.show()

if __name__ == "__main__":
    main()
