# modified from SynRS3D
# https://github.com/JTRNEO/SynRS3D/blob/main/prepare_ISPRS_datasets/crop_isprs.py

import os
import argparse
import numpy as np
from osgeo import gdal

# Color palette mapping (only used if --label is enabled)
palette = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5       # Clutter/background
}

def convert_label(img_hwc: np.ndarray) -> np.ndarray:
    """Convert RGB label image (H,W,3) into single-channel class IDs (H,W)."""
    single_channel_label = np.zeros((img_hwc.shape[0], img_hwc.shape[1]), dtype=np.uint8)
    for key, value in palette.items():
        mask = np.all(img_hwc == np.array(key, dtype=img_hwc.dtype), axis=-1)
        single_channel_label[mask] = value
    return single_channel_label

def process_image_gdal(img_path: str, slice_size: int, is_label: bool = False):
    print(f"\n[DEBUG] Processing: {img_path}")
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {img_path}")

    print("[DEBUG] GDAL dataset opened.")
    print(f"[DEBUG] RasterXSize: {ds.RasterXSize}, RasterYSize: {ds.RasterYSize}, BandCount: {ds.RasterCount}")

    img = ds.ReadAsArray()
    if img is None:
        raise RuntimeError(f"ReadAsArray() returned None for: {img_path}")

    print(f"[DEBUG] ReadAsArray() completed. Image shape: {img.shape}")
    print(f"[DEBUG] Data type: {img.dtype}, Min: {np.nanmin(img)}, Max: {np.nanmax(img)}")

    original_geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # Normalize to (C,H,W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    img = np.nan_to_num(img)

    # Optional label conversion (RGB -> single channel)
    if is_label and img.shape[0] == 3:
        img_hwc = np.transpose(img, (1, 2, 0))
        img_single = convert_label(img_hwc)  # (H,W)
        img = img_single[np.newaxis, ...]

    height, width = img.shape[1], img.shape[2]

    num_slices_height = (height - 1) // slice_size + 1
    num_slices_width = (width - 1) // slice_size + 1
    step_height = (height - slice_size) // (num_slices_height - 1) if num_slices_height > 1 else slice_size
    step_width = (width - slice_size) // (num_slices_width - 1) if num_slices_width > 1 else slice_size

    print(f"[DEBUG] Image dimensions: {height}x{width}")
    print(f"[DEBUG] Will create {num_slices_height} x {num_slices_width} = {num_slices_height * num_slices_width} slices")

    slices = []
    slice_geotransforms = []

    gt = original_geotransform
    # Correct for rotated geotransforms too (handles gt[2], gt[4])
    for i in range(num_slices_height):
        for j in range(num_slices_width):
            start_row = min(i * step_height, height - slice_size)
            start_col = min(j * step_width, width - slice_size)

            slice_img = img[:, start_row:start_row + slice_size, start_col:start_col + slice_size]

            new_geotransform = (
                gt[0] + start_col * gt[1] + start_row * gt[2],
                gt[1],
                gt[2],
                gt[3] + start_col * gt[4] + start_row * gt[5],
                gt[4],
                gt[5],
            )

            slices.append(slice_img)
            slice_geotransforms.append(new_geotransform)

    return slices, slice_geotransforms, projection, ds

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def gdal_dtype_from_dataset(ds: gdal.Dataset) -> int:
    """Use dataset's first band data type as output type."""
    b = ds.GetRasterBand(1)
    return b.DataType if b is not None else gdal.GDT_Byte

def crop_directory(input_dir: str, output_dir: str, slice_size: int, is_label: bool = False):
    ensure_dir(output_dir)

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("Could not get GTiff driver")

    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith("."):
            continue

        in_path = os.path.join(input_dir, filename)
        if not os.path.isfile(in_path):
            continue

        slices, slice_geotransforms, projection, src_ds = process_image_gdal(
            in_path, slice_size=slice_size, is_label=is_label
        )

        out_dtype = gdal_dtype_from_dataset(src_ds)
        # If label conversion is enabled, we force Byte output (class ids)
        if is_label:
            out_dtype = gdal.GDT_Byte

        print(f"[DEBUG] Saving {len(slices)} slices to: {output_dir}")

        base = os.path.splitext(filename)[0]
        for idx, (slice_img, geotransform) in enumerate(zip(slices, slice_geotransforms)):
            out_name = f"{base}_slice_{idx}.tif"
            out_path = os.path.join(output_dir, out_name)

            channels = slice_img.shape[0]
            out_ds = driver.Create(out_path, slice_size, slice_size, channels, out_dtype)
            if out_ds is None:
                raise RuntimeError(f"Could not create output: {out_path}")

            out_ds.SetGeoTransform(geotransform)
            if projection:
                out_ds.SetProjection(projection)

            for ch in range(channels):
                out_band = out_ds.GetRasterBand(ch + 1)
                out_band.WriteArray(slice_img[ch, :, :])
                out_band.FlushCache()

            out_ds = None  # close

def main():
    parser = argparse.ArgumentParser(
        description="Crop rasters into overlapping 256x256 (or configurable) tiles, preserving georeferencing when present."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="../data/",
        help="Base input path. Default: ../data/ . Expected subdirs raw_image/ and raw_dsm/ (and optionally raw_label/ if used).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../data/",
        help="Base output path. Default: ../data/ . Outputs go to image_cropped/ and dsm_cropped/ (and label_cropped/ if used).",
    )
    parser.add_argument(
        "--slice-size",
        "-s",
        type=int,
        default=256,
        help="Tile size in pixels (square). Default: 256",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Also process raw_label/ as RGB palette labels -> single-channel class ids.",
    )

    args = parser.parse_args()

    in_base = os.path.abspath(args.input)
    out_base = os.path.abspath(args.output)

    raw_image = os.path.join(in_base, "raw_image")
    raw_dsm = os.path.join(in_base, "raw_dsm")

    out_image = os.path.join(out_base, "image_cropped")
    out_dsm = os.path.join(out_base, "dsm_cropped")

    if not os.path.isdir(raw_image):
        raise SystemExit(f"Missing input directory: {raw_image}")
    if not os.path.isdir(raw_dsm):
        raise SystemExit(f"Missing input directory: {raw_dsm}")

    print(f"[INFO] Input base:  {in_base}")
    print(f"[INFO] Output base: {out_base}")
    print(f"[INFO] Slice size:  {args.slice_size}")

    print("\n[INFO] Cropping raw_image -> image_cropped")
    crop_directory(raw_image, out_image, slice_size=args.slice_size, is_label=False)

    print("\n[INFO] Cropping raw_dsm -> dsm_cropped")
    crop_directory(raw_dsm, out_dsm, slice_size=args.slice_size, is_label=False)

    if args.label:
        raw_label = os.path.join(in_base, "raw_label")
        out_label = os.path.join(out_base, "label_cropped")
        if not os.path.isdir(raw_label):
            raise SystemExit(f"--label was set but missing input directory: {raw_label}")

        print("\n[INFO] Cropping raw_label -> label_cropped (palette -> class ids)")
        crop_directory(raw_label, out_label, slice_size=args.slice_size, is_label=True)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()