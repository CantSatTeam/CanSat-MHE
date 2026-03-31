"""
Slice a large DEM/DSM GeoTIFF into fixed-size square tiles defined by:

  - pixel dimensions (tile_px x tile_px)
  - target ground sample distance / resolution (gsd meters per pixel)

Defaults:
  input_dir  = ../data/dsm/input
  output_dir = ../data/dsm/output

Each tile:
  - is tile_px x tile_px pixels
  - is written at exactly `gsd` meters / pixel
  - is georeferenced (GeoTransform + Projection)
  - is written in a projected CRS (UTM) so sizes are truly metric

Outputs:
  <output_dir>/<stem>_tile_<row>_<col>.tif

Usage:
  python slice.py
  python slice.py --tile_px 1024 --gsd 0.5
  python slice.py --input_dir ../data/dsm/input --output_dir ../data/dsm/output --tile_px 1024 --gsd 0.5 --overlap_px 128
  python slice.py my_raster.tif --single --tile_px 1024 --gsd 0.5

Notes:
  - Requires GDAL CLI: gdalwarp
"""

import sys
import math
import subprocess
from pathlib import Path

import rasterio
import rasterio.warp


DEFAULT_INPUT_DIR = Path("../data/dsm/input")
DEFAULT_OUTPUT_DIR = Path("../data/dsm/output")


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def snap_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def parse_args(argv: list[str]) -> dict:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    tile_px = 1024
    overlap_px = 0
    gsd = 1.0
    specific_file = None
    single_mode = False

    i = 1

    # positional filename still supported
    if i < len(argv) and not argv[i].startswith("--"):
        specific_file = argv[i]
        i += 1

    while i < len(argv):
        a = argv[i]
        if a == "--input_dir":
            input_dir = Path(argv[i + 1])
            i += 2
        elif a == "--output_dir":
            output_dir = Path(argv[i + 1])
            i += 2
        elif a == "--tile_px":
            tile_px = int(argv[i + 1])
            i += 2
        elif a == "--overlap_px":
            overlap_px = int(argv[i + 1])
            i += 2
        elif a == "--gsd":
            gsd = float(argv[i + 1])
            i += 2
        elif a == "--single":
            single_mode = True
            i += 1
        else:
            raise ValueError(f"Unknown arg: {a}")

    if tile_px <= 0:
        raise ValueError("--tile_px must be > 0")
    if gsd <= 0:
        raise ValueError("--gsd must be > 0")
    if overlap_px < 0:
        raise ValueError("--overlap_px must be >= 0")
    if overlap_px >= tile_px:
        raise ValueError("--overlap_px must be smaller than --tile_px")

    return {
        "input_dir": input_dir.resolve(),
        "output_dir": output_dir.resolve(),
        "tile_px": tile_px,
        "overlap_px": overlap_px,
        "gsd": gsd,
        "specific_file": specific_file,
        "single_mode": single_mode,
    }


def compute_utm_for_dataset(tif_path: Path) -> str:
    """Choose a UTM CRS based on the dataset center."""
    with rasterio.open(tif_path) as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.bottom + ds.bounds.top) / 2.0
        lon, lat = rasterio.warp.transform(ds.crs, "EPSG:4326", [cx], [cy])
        lon, lat = lon[0], lat[0]
    epsg = utm_epsg_from_lonlat(lon, lat)
    return f"EPSG:{epsg}"


def dataset_bounds_in_utm(tif_path: Path, utm_srs: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of dataset in utm_srs meters."""
    with rasterio.open(tif_path) as ds:
        b = ds.bounds
        crs = ds.crs

    xs = [b.left, b.right, b.right, b.left]
    ys = [b.bottom, b.bottom, b.top, b.top]
    xutm, yutm = rasterio.warp.transform(crs, utm_srs, xs, ys)

    xmin, xmax = min(xutm), max(xutm)
    ymin, ymax = min(yutm), max(yutm)
    return xmin, ymin, xmax, ymax


def tile_one_raster(in_path: Path, out_dir: Path, tile_px: int, overlap_px: int, gsd: float) -> int:
    """
    Tile raster into exact tile_px x tile_px tiles at the requested gsd.
    """
    utm_srs = compute_utm_for_dataset(in_path)
    xmin, ymin, xmax, ymax = dataset_bounds_in_utm(in_path, utm_srs)

    tile_size_m = tile_px * gsd
    step_px = tile_px - overlap_px
    step_m = step_px * gsd

    # Deterministic grid alignment
    gx0 = snap_down(xmin, step_m)
    gy0 = snap_down(ymin, step_m)

    ncols = int(math.ceil((xmax - gx0) / step_m))
    nrows = int(math.ceil((ymax - gy0) / step_m))

    stem = in_path.stem
    print(f"\n[INFO] Tiling: {in_path.name}")
    print(f"[INFO] CRS: {utm_srs}")
    print(f"[INFO] Tile size: {tile_px} px x {tile_px} px")
    print(f"[INFO] GSD: {gsd} m/px")
    print(f"[INFO] Overlap: {overlap_px} px (step={step_px} px / {step_m} m)")
    print(f"[INFO] Ground tile size: {tile_size_m} m x {tile_size_m} m")
    print(f"[INFO] UTM bounds: xmin={xmin:.3f}, ymin={ymin:.3f}, xmax={xmax:.3f}, ymax={ymax:.3f}")
    print(f"[INFO] Grid: {nrows} rows x {ncols} cols")

    written = 0
    for r in range(nrows):
        for c in range(ncols):
            x0 = gx0 + c * step_m
            y0 = gy0 + r * step_m
            x1 = x0 + tile_size_m
            y1 = y0 + tile_size_m

            # Skip tiles fully outside
            if x1 <= xmin or x0 >= xmax or y1 <= ymin or y0 >= ymax:
                continue

            out_tile = out_dir / f"{stem}_tile_{r:04d}_{c:04d}.tif"

            run([
                "gdalwarp",
                "-t_srs", utm_srs,
                "-te", f"{x0:.12f}", f"{y0:.12f}", f"{x1:.12f}", f"{y1:.12f}",
                "-tr", f"{gsd:.12f}", f"{gsd:.12f}",
                "-r", "bilinear",
                "-ot", "Float32",
                "-dstnodata", "0",
                "-overwrite",
                str(in_path),
                str(out_tile),
            ])
            written += 1

    print(f"[DONE] {in_path.name}: wrote {written} tiles")
    return written


def crop_single_tile(in_path: Path, out_dir: Path, tile_px: int, gsd: float) -> Path:
    """
    Crops a single tile (tile_px x tile_px) centered on the dataset.
    """
    utm_srs = compute_utm_for_dataset(in_path)
    xmin, ymin, xmax, ymax = dataset_bounds_in_utm(in_path, utm_srs)

    tile_size_m = tile_px * gsd

    # Center in UTM meters
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    x0 = snap_down(cx - tile_size_m / 2.0, gsd)
    y0 = snap_down(cy - tile_size_m / 2.0, gsd)
    x1 = x0 + tile_size_m
    y1 = y0 + tile_size_m

    stem = in_path.stem
    out_tile = out_dir / f"{stem}_single_{tile_px}px_{str(gsd).replace('.', 'p')}mpp.tif"

    print(f"\n[INFO] Cropping SINGLE tile from: {in_path.name}")
    print(f"[INFO] CRS: {utm_srs}")
    print(f"[INFO] Tile size: {tile_px} px x {tile_px} px")
    print(f"[INFO] GSD: {gsd} m/px")
    print(f"[INFO] Crop bounds: xmin={x0:.3f}, ymin={y0:.3f}, xmax={x1:.3f}, ymax={y1:.3f}")

    run([
        "gdalwarp",
        "-t_srs", utm_srs,
        "-te", f"{x0:.12f}", f"{y0:.12f}", f"{x1:.12f}", f"{y1:.12f}",
        "-tr", f"{gsd:.12f}", f"{gsd:.12f}",
        "-r", "bilinear",
        "-ot", "Float32",
        "-dstnodata", "0",
        "-overwrite",
        str(in_path),
        str(out_tile),
    ])

    print(f"[DONE] Wrote single tile: {out_tile}")
    return out_tile


def main() -> None:
    args = parse_args(sys.argv)
    input_dir: Path = args["input_dir"]
    output_dir: Path = args["output_dir"]
    tile_px: int = args["tile_px"]
    overlap_px: int = args["overlap_px"]
    gsd: float = args["gsd"]
    single_mode: bool = args["single_mode"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    specific_file = args["specific_file"]

    # Build list of input tifs
    if specific_file is not None:
        tif_path = input_dir / specific_file
        if not tif_path.exists():
            raise FileNotFoundError(f"{tif_path} not found")
        tifs = [tif_path]
    else:
        tifs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".tif", ".tiff"]])

    if not tifs:
        raise RuntimeError(f"No .tif/.tiff files found in {input_dir}")

    if single_mode:
        written = 0
        for tif in tifs:
            crop_single_tile(tif, output_dir, tile_px, gsd)
            written += 1
        print(f"\n[ALL DONE] Wrote {written} single crop(s)")
        print(f"[ALL DONE] Output dir: {output_dir}")
    else:
        total_tiles = 0
        for tif in tifs:
            total_tiles += tile_one_raster(tif, output_dir, tile_px, overlap_px, gsd)
        print(f"\n[ALL DONE] Total tiles written: {total_tiles}")
        print(f"[ALL DONE] Output dir: {output_dir}")


if __name__ == "__main__":
    main()