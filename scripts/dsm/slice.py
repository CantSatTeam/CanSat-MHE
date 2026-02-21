"""
Slice (tile) a large DEM/DSM GeoTIFF into fixed physical-area square tiles.

Defaults:
  input_dir  = ../data/dsm/input
  output_dir = ../data/dsm/output

Each tile:
  - is tile_size_m x tile_size_m meters on the ground
  - is georeferenced (GeoTransform + Projection)
  - is written in a projected CRS (UTM) so sizes are truly metric

Outputs:
  <output_dir>/<stem>_tile_<row>_<col>.tif

Usage:
  python slice.py
  python slice.py --tile_size_m 1000
  python slice.py --input_dir ../data/dsm/input --output_dir ../data/dsm/output --tile_size_m 1000 --overlap_m 200

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


def parse_args(argv: list[str]) -> dict:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    tile_size_m = 1000.0
    overlap_m = 0.0
    specific_file = None

    i = 1

    # If first argument exists and is not a flag, treat it as DSM filename
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
        elif a == "--tile_size_m":
            tile_size_m = float(argv[i + 1])
            i += 2
        elif a == "--overlap_m":
            overlap_m = float(argv[i + 1])
            i += 2
        else:
            raise ValueError(f"Unknown arg: {a}")

    if overlap_m < 0 or overlap_m >= tile_size_m:
        raise ValueError("overlap_m must be in [0, tile_size_m).")

    return {
        "input_dir": input_dir.resolve(),
        "output_dir": output_dir.resolve(),
        "tile_size_m": tile_size_m,
        "overlap_m": overlap_m,
        "specific_file": specific_file,
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


def tile_one_raster(in_path: Path, out_dir: Path, tile_size_m: float, overlap_m: float) -> int:
    utm_srs = compute_utm_for_dataset(in_path)
    xmin, ymin, xmax, ymax = dataset_bounds_in_utm(in_path, utm_srs)

    step = tile_size_m - overlap_m

    # Deterministic grid alignment
    gx0 = math.floor(xmin / step) * step
    gy0 = math.floor(ymin / step) * step

    ncols = int(math.ceil((xmax - gx0) / step))
    nrows = int(math.ceil((ymax - gy0) / step))

    stem = in_path.stem
    print(f"\n[INFO] Tiling: {in_path.name}")
    print(f"[INFO] CRS: {utm_srs}")
    print(f"[INFO] Tile size: {tile_size_m} m, overlap: {overlap_m} m (step={step} m)")
    print(f"[INFO] UTM bounds: xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
    print(f"[INFO] Grid: {nrows} rows x {ncols} cols")

    written = 0
    for r in range(nrows):
        for c in range(ncols):
            x0 = gx0 + c * step
            y0 = gy0 + r * step
            x1 = x0 + tile_size_m
            y1 = y0 + tile_size_m

            # Skip tiles fully outside
            if x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax:
                continue

            out_tile = out_dir / f"{stem}_tile_{r:04d}_{c:04d}.tif"

            run([
                "gdalwarp",
                "-t_srs", utm_srs,
                "-te", str(x0), str(y0), str(x1), str(y1),
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


def main() -> None:
    args = parse_args(sys.argv)
    input_dir: Path = args["input_dir"]
    output_dir: Path = args["output_dir"]
    tile_size_m: float = args["tile_size_m"]
    overlap_m: float = args["overlap_m"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    specific_file = args["specific_file"]

    if specific_file is not None:
        tif_path = input_dir / specific_file
        if not tif_path.exists():
            raise FileNotFoundError(f"{tif_path} not found")
        tifs = [tif_path]
    else:
        tifs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".tif", ".tiff"]])

    if not tifs:
        raise RuntimeError(f"No .tif/.tiff files found in {input_dir}")

    total_tiles = 0
    for tif in tifs:
        total_tiles += tile_one_raster(tif, output_dir, tile_size_m, overlap_m)

    print(f"\n[ALL DONE] Total tiles written: {total_tiles}")
    print(f"[ALL DONE] Output dir: {output_dir}")


if __name__ == "__main__":
    main()