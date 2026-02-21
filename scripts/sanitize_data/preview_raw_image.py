from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import os

# Single image
# filepath = "image_cropped/TOP_Mosaic_09cm_slice_784.tif"

# Load the huge geotiff by creating a downsampled preview
raw_image = "raw_image/TOP_Mosaic_09cm.tif"
preview_path = "raw_image_preview.tif"

print(f"Creating downsampled preview of {raw_image}...")
subprocess.run([
    'gdal_translate',
    '-outsize', '25%', '25%',  # Scale to 25% to avoid decompression bomb
    raw_image,
    preview_path
], check=True, capture_output=True)

print(f"Loading preview...")
img_pil = Image.open(preview_path)
arr = np.array(img_pil)

print(f"Preview shape: {arr.shape}")

nir = arr[:,:,0].astype(np.float32) / 255.0
red = arr[:,:,1].astype(np.float32) / 255.0
green = arr[:,:,2].astype(np.float32) / 255.0

print("Creating all 6 blue synthesis methods...\n")

# All 6 blue synthesis methods
blue_method1 = np.clip(green - np.maximum(0, nir - red) * 0.5, 0, 1)

veg_index = (nir - red) / (nir + red + 1e-6)
blue_method2 = np.clip(green * (1 - veg_index * 0.3), 0, 1)

blue_method3 = np.clip(green * 0.9 - red * 0.1, 0, 1)

ratio = red / (green + 1e-6)
blue_method4 = np.clip(green / (1 + ratio * 0.5), 0, 1)

blue_method5 = np.clip(green * 0, 0, 1)

blue_method6 = np.minimum(red, green)

methods = [
    np.stack([red, green, blue_method1], axis=2),
    np.stack([red, green, blue_method2], axis=2),
    np.stack([red, green, blue_method3], axis=2),
    np.stack([red, green, blue_method4], axis=2),
    np.stack([red, green, blue_method5], axis=2),
    np.stack([red, green, blue_method6], axis=2),
]

titles = [
    "NIR-Constrained",
    "Inverse NIR (CHOSEN)",
    "Histogram-Match",
    "Red-Green Ratio",
    "Blue = 0",
    "Min(R,G)"
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (img_array, title) in enumerate(zip(methods, titles)):
    ax = axes[idx // 3, idx % 3]
    ax.imshow(np.clip(img_array, 0, 1))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig("output_raw_dataset_preview.png", dpi=100, bbox_inches='tight')
print(f"Saved raw dataset preview (25% scale) to output_raw_dataset_preview.png")
