import argparse
import hashlib
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


TIFF_SUFFIXES = {".tif", ".tiff"}
PIL_SUFFIXES = {".png", ".jpg", ".jpeg"}
IMAGE_SUFFIXES = TIFF_SUFFIXES | PIL_SUFFIXES


def is_tiff(path: Path) -> bool:
    return path.suffix.lower() in TIFF_SUFFIXES


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not is_supported_image(input_path):
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and is_supported_image(p)
        )
        if not files:
            raise ValueError(f"No supported image files found in directory: {input_path}")
        return files

    raise ValueError(f"Input path does not exist: {input_path}")


def resolve_output_file(input_root: Path, input_file: Path, output_path: Path) -> Path:
    # Single-file input: output can be either an image file path or a directory.
    if input_root.is_file():
        if output_path.suffix.lower() in IMAGE_SUFFIXES:
            output_file = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / input_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file

    # Directory input: output is treated as a directory and mirrors structure.
    rel = input_file.relative_to(input_root)
    output_file = output_path / rel
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


def load_depth(input_file: Path) -> np.ndarray:
    suffix = input_file.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        depth = tifffile.imread(input_file)
    elif suffix in PIL_SUFFIXES:
        with Image.open(input_file) as img:
            depth = np.array(img)
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    if depth.ndim < 2:
        raise ValueError(f"Expected an image with at least 2 dimensions, got shape {depth.shape}")

    # For common image formats, reject multi-channel RGB/RGBA depth inputs.
    if suffix in PIL_SUFFIXES and depth.ndim != 2:
        raise ValueError(
            f"Expected a single-channel PNG/JPG depth image, got shape {depth.shape} for {input_file}"
        )

    return depth


def prepare_array_for_png(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"PNG output expects a single-channel 2D image, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError("PNG output for floating-point depth is not supported here. Use TIFF instead.")

    if arr.dtype == np.uint8 or arr.dtype == np.uint16:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        min_val = int(arr.min())
        max_val = int(arr.max())
        if min_val >= 0 and max_val <= 255:
            return arr.astype(np.uint8)
        if min_val >= 0 and max_val <= 65535:
            return arr.astype(np.uint16)

    raise ValueError(
        f"PNG output requires uint8/uint16-compatible integer data, got dtype {arr.dtype}"
    )


def prepare_array_for_jpeg(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"JPEG output expects a single-channel 2D image, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        raise ValueError("JPEG output for floating-point depth is not supported. Use TIFF or PNG instead.")

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        min_val = int(arr.min())
        max_val = int(arr.max())
        if min_val >= 0 and max_val <= 255:
            return arr.astype(np.uint8)

    raise ValueError(
        f"JPEG output requires uint8-compatible integer data in [0, 255], got dtype {arr.dtype}"
    )


def save_depth(output_file: Path, depth: np.ndarray) -> None:
    suffix = output_file.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        tifffile.imwrite(output_file, depth)
        return

    if suffix == ".png":
        Image.fromarray(prepare_array_for_png(depth)).save(output_file)
        return

    if suffix in {".jpg", ".jpeg"}:
        jpeg_arr = prepare_array_for_jpeg(depth)
        Image.fromarray(jpeg_arr).save(output_file, quality=100, subsampling=0)
        return

    raise ValueError(f"Unsupported output format: {output_file}")


def sample_sparse_depth(depth: np.ndarray, points: int, rng: np.random.Generator) -> np.ndarray:
    if depth.ndim < 2:
        raise ValueError(f"Expected a depth map with at least 2 dimensions, got shape {depth.shape}")

    valid_mask = np.isfinite(depth) & (depth > 0)
    valid_indices = np.flatnonzero(valid_mask)

    if len(valid_indices) == 0:
        raise ValueError("No valid depth pixels found. Expected finite values > 0.")

    n = min(points, len(valid_indices))
    chosen = rng.choice(valid_indices, size=n, replace=False)

    sparse = np.zeros_like(depth)
    flat_sparse = sparse.reshape(-1)
    flat_depth = depth.reshape(-1)
    flat_sparse[chosen] = flat_depth[chosen]
    return sparse


def stable_path_seed(path: Path) -> int:
    digest = hashlib.blake2b(
        str(path.resolve()).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def process_file(input_file: Path, output_file: Path, points: int, seed: int | None) -> None:
    depth = load_depth(input_file)

    # Make per-file sampling reproducible while still varying across files.
    if seed is None:
        rng = np.random.default_rng()
    else:
        file_seed = (seed + stable_path_seed(input_file)) % (2**63 - 1)
        rng = np.random.default_rng(file_seed)

    sparse = sample_sparse_depth(depth, points, rng)
    save_depth(output_file, sparse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a sparse depth image by keeping only N valid depth pixels."
    )
    parser.add_argument(
        "input_path",
        help="Input image file or directory containing .tif/.tiff/.png/.jpg/.jpeg files.",
    )
    parser.add_argument(
        "output_path",
        help="Output image file or directory.",
    )
    parser.add_argument(
        "--points",
        type=int,
        required=True,
        help="Number of valid depth pixels to keep in each output image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.points <= 0:
        raise ValueError("--points must be a positive integer.")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    input_files = collect_input_files(input_path)

    for input_file in input_files:
        output_file = resolve_output_file(input_path, input_file, output_path)
        process_file(input_file, output_file, args.points, args.seed)
        print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()