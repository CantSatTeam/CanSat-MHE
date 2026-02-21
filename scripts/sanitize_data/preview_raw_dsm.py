from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import os

# Load the huge DSM geotiff by creating a downsampled preview
raw_dsm = "raw_dsm/DSM_09cm_matching.tif"
preview_path = "raw_dsm_preview.tif"

print(f"Creating downsampled preview of {raw_dsm}...")
subprocess.run([
    'gdal_translate',
    '-outsize', '25%', '25%',  # Scale to 25% to avoid decompression bomb
    raw_dsm,
    preview_path
], check=True, capture_output=True)

print(f"Loading DSM preview...")
img_pil = Image.open(preview_path)
arr = np.array(img_pil)

print(f"Preview shape: {arr.shape}")
print(f"DSM dtype: {arr.dtype}")
print(f"Height range: min={arr.min():.2f}m, max={arr.max():.2f}m, mean={arr.mean():.2f}m")
print(f"Height std: {arr.std():.2f}m")
print(f"Percentiles: 1%={np.percentile(arr, 1):.2f}, 99%={np.percentile(arr, 99):.2f}")

# Clip outliers for better visualization
vmin = np.percentile(arr, 1)
vmax = np.percentile(arr, 99)
arr_clipped = np.clip(arr, vmin, vmax)
print(f"Clipped range: {vmin:.2f}m to {vmax:.2f}m")

# Create multiple visualizations of the DSM
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Grayscale (clipped)
axes[0, 0].imshow(arr_clipped, cmap='gray', vmin=vmin, vmax=vmax)
axes[0, 0].set_title('Grayscale (Clipped 1-99%)', fontweight='bold')
axes[0, 0].axis('off')

# 2. Terrain colormap (clipped)
axes[0, 1].imshow(arr_clipped, cmap='terrain', vmin=vmin, vmax=vmax)
axes[0, 1].set_title('Terrain (Clipped)', fontweight='bold')
axes[0, 1].axis('off')

# 3. Viridis (clipped with colorbar)
im3 = axes[0, 2].imshow(arr_clipped, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0, 2].set_title('Viridis (Clipped)', fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

# 4. Hillshade (3D effect)
# Calculate hillshade for 3D visualization
# Assuming sun angle of 45° from NW
from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(arr, vert_exag=1.5)
axes[1, 0].imshow(hillshade, cmap='gray')
axes[1, 0].set_title('Hillshade (3D Effect)', fontweight='bold')
axes[1, 0].axis('off')

# 5. Hot colormap (clipped)
axes[1, 1].imshow(arr_clipped, cmap='hot', vmin=vmin, vmax=vmax)
axes[1, 1].set_title('Hot (Clipped)', fontweight='bold')
axes[1, 1].axis('off')

# 6. Normalized to clipped range
arr_norm = (arr_clipped - vmin) / (vmax - vmin)
axes[1, 2].imshow(arr_norm, cmap='coolwarm')
axes[1, 2].set_title('Normalized Cool-Warm (Clipped)', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("output_raw_dsm_preview.png", dpi=100, bbox_inches='tight')
print(f"\nSaved raw DSM preview (25% scale) to output_raw_dsm_preview.png")
