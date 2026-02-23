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

print("Creating Inverse NIR blue band...\n")

# Inverse NIR
veg_index = (nir - red) / (nir + red + 1e-6)
blue_method2 = np.clip(green * (1 - veg_index * 0.3), 0, 1)

img_array = np.stack([red, green, blue_method2], axis=2)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(np.clip(img_array, 0, 1))
ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig("output_raw_dataset_preview.png", dpi=100, bbox_inches='tight', pad_inches=0)
print(f"Saved raw dataset preview (25% scale) to output_raw_dataset_preview.png")
