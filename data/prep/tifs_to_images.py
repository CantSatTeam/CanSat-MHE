#!/usr/bin/env python3
"""
Convert TIF files to standard image formats (PNG/JPG).

Handles two types:
1. Heightmaps/DSM - single band TIFs converted to grayscale
2. IR-R-G TIFs - 3-band TIFs (NIR-Red-Green) converted to RGB using vegetation index

Usage:
    python convert_tifs_to_images.py <input_dir> <output_dir> [--format png|jpg]
    python convert_tifs_to_images.py datasets/temp/1_DSM output_images --format png
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def is_heightmap(arr: np.ndarray) -> bool:
    """Detect if this is a single-band heightmap/DSM."""
    if len(arr.shape) == 2:
        return True
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        return True
    return False


def convert_heightmap_to_grayscale(arr: np.ndarray) -> Image.Image:
    """Convert heightmap to normalized grayscale image."""
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]
    
    # Normalize to 0-255
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max > arr_min:
        normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(arr, dtype=np.uint8)
    
    return Image.fromarray(normalized, mode='L')


def convert_rgb_direct(arr: np.ndarray) -> Image.Image:
    """
    Convert 3-band RGB TIF directly to RGB image.
    """
    # Just pass through as-is
    if arr.dtype != np.uint8:
        # Normalize if needed
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr, mode='RGB')


def convert_irrg_to_rgb(arr: np.ndarray) -> Image.Image:
    """
    Convert IR-R-G (NIR-Red-Green) to RGB using vegetation index.
    
    Input bands:
        - Band 0: NIR (Near-Infrared)
        - Band 1: Red
        - Band 2: Green
    
    Output:
        - R: Red
        - G: Green
        - B: Synthetic blue from vegetation index
    """
    nir = arr[:, :, 0].astype(np.float32) / 255.0
    red = arr[:, :, 1].astype(np.float32) / 255.0
    green = arr[:, :, 2].astype(np.float32) / 255.0
    
    # Vegetation index: (NIR - Red) / (NIR + Red)
    veg_index = (nir - red) / (nir + red + 1e-6)
    
    # Create synthetic blue band using inverse NIR method
    blue = np.clip(green * (1 - veg_index * 0.3), 0, 1)
    
    # Stack as RGB and convert to uint8
    rgb = np.stack([red, green, blue], axis=2)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(rgb, mode='RGB')


def convert_tif(tif_path: Path, output_dir: Path, output_format: str = 'png', use_nir: bool = False) -> bool:
    """
    Convert a single TIF file to image format.
    
    Returns True on success, False on failure.
    """
    try:
        # Load TIF
        img = Image.open(tif_path)
        arr = np.array(img)
        
        # Determine type and convert
        if is_heightmap(arr):
            output_img = convert_heightmap_to_grayscale(arr)
        elif len(arr.shape) == 3 and arr.shape[2] == 3:
            if use_nir:
                output_img = convert_irrg_to_rgb(arr)
            else:
                output_img = convert_rgb_direct(arr)
        else:
            print(f"WARNING: Unknown format for {tif_path.name}: shape={arr.shape}")
            return False
        
        # Save output
        output_path = output_dir / f"{tif_path.stem}.{output_format}"
        
        if output_format == 'jpg':
            output_img.save(output_path, quality=95)
        else:
            output_img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to convert {tif_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert TIF files to PNG/JPG images'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing TIF files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save converted images'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'jpg'],
        default='png',
        help='Output image format (default: png)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='File pattern to match (default: *.tif)'
    )
    parser.add_argument(
        '--nir',
        action='store_true',
        help='Apply NIR to RGB conversion using vegetation index (for IR-R-G TIFs)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    tif_files = sorted(input_dir.glob(args.pattern))
    
    if not tif_files:
        print(f"ERROR: No TIF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(tif_files)} TIF file(s)")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.format.upper()}\n")
    
    # Convert all files
    success_count = 0
    failed_count = 0
    
    for i, tif_path in enumerate(tif_files, 1):
        print(f"[{i}/{len(tif_files)}] {tif_path.name}...", end=' ')
        if convert_tif(tif_path, output_dir, args.format, args.nir):
            print("OK")
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print(f"\nSuccessfully converted: {success_count}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
