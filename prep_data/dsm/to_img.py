"""
Download matching Sentinel-2 RGB for DSM/DEM tile GeoTIFFs produced by slice.py

Defaults:
  input_dir  = ../data/dsm/output      (DSM tiles)
  output_dir = ../data/dsm/output      (RGB saved alongside tiles by default)

Outputs (for each tile <tile>.tif):
  <output_dir>/<tile>.jpg
  <output_dir>/<tile>_rgb.tif          (NOT georeferenced; pixel-aligned only)

Usage:
  python to_img.py
  python to_img.py --input_dir ../data/dsm/output --output_dir ../data/dsm/output
  python to_img.py --start 2023-06-01 --end 2023-09-01 --maxcloud 30
  python to_img.py --pattern "*_tile_*.tif" --skip_existing 1

Notes:
  - Requires Earth Engine auth + project set (ee.Initialize() must work)
  - Sentinel-2 RGB native resolution is 10 m (B2/B3/B4)
  - This writes RGB TIFFs without georeferencing; if you want georeferenced RGB,
    say so and I’ll switch to a GDAL warp-to-match pipeline.
"""

import sys
import zipfile
from pathlib import Path

import ee
import requests
import rasterio
import rasterio.warp
import numpy as np
from PIL import Image


DEFAULT_INPUT_DIR = Path("../data/dsm/input")
DEFAULT_OUTPUT_DIR = Path("../data/dsm/output")

SATELLITE_SR = "COPERNICUS/S2_SR_HARMONIZED"
RGB_BANDS = ["B4", "B3", "B2"]

S2_RGB_SCALE_M = 10  # Sentinel-2 native RGB resolution

MINN = 0
MAXX = 3000


def parse_args(argv: list[str]) -> dict:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    start = "2023-06-01"
    end = "2023-09-01"
    maxcloud = 30.0
    pattern = "*_tile_*.tif"
    skip_existing = True
    specific_file = None

    i = 1

    # If first argument exists and is not a flag, treat it as a specific DSM tile filename
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
        elif a == "--start":
            start = argv[i + 1]
            i += 2
        elif a == "--end":
            end = argv[i + 1]
            i += 2
        elif a == "--maxcloud":
            maxcloud = float(argv[i + 1])
            i += 2
        elif a == "--pattern":
            pattern = argv[i + 1]
            i += 2
        elif a == "--skip_existing":
            skip_existing = bool(int(argv[i + 1]))
            i += 2
        else:
            raise ValueError(f"Unknown arg: {a}")

    return {
        "input_dir": input_dir.resolve(),
        "output_dir": output_dir.resolve(),
        "start": start,
        "end": end,
        "maxcloud": maxcloud,
        "pattern": pattern,
        "skip_existing": skip_existing,
        "specific_file": specific_file,
    }


def bounds_polygon_4326(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as ds:
        b = ds.bounds
        crs = ds.crs

    xs = [b.left, b.right, b.right, b.left, b.left]
    ys = [b.bottom, b.bottom, b.top, b.top, b.bottom]
    lons, lats = rasterio.warp.transform(crs, "EPSG:4326", xs, ys)

    coords = [[[float(lons[i]), float(lats[i])] for i in range(len(lons))]]
    return {"type": "Polygon", "coordinates": coords}


def download_s2_rgb(region_geojson_4326: dict, tempdir: Path, start: str, end: str, maxcloud: float) -> Path:
    ee_geom = ee.Geometry(region_geojson_4326)

    coll = (ee.ImageCollection(SATELLITE_SR)
            .filterBounds(ee_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", maxcloud))
            .select(RGB_BANDS))

    if coll.size().getInfo() == 0:
        print("[WARN] No images in date/cloud filter. Falling back to all dates for this region.")
        coll = ee.ImageCollection(SATELLITE_SR).filterBounds(ee_geom).select(RGB_BANDS)

    img = coll.median()

    url = img.getDownloadURL({
        "region": ee_geom,
        "scale": S2_RGB_SCALE_M,
        "crs": "EPSG:4326",
        "filePerBand": True,
        "format": "GEO_TIFF",
    })

    out_bin = tempdir / "gee_rgb.bin"
    if out_bin.exists():
        out_bin.unlink()

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_bin, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return out_bin


def extract_tifs(out_bin: Path, tempdir: Path) -> list[Path]:
    if zipfile.is_zipfile(out_bin):
        zip_path = tempdir / "gee_rgb.zip"
        if zip_path.exists():
            zip_path.unlink()
        out_bin.rename(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tempdir)

        zip_path.unlink(missing_ok=True)
        tifs = list(tempdir.glob("*.tif"))
    else:
        tif_path = tempdir / "gee_rgb.tif"
        if tif_path.exists():
            tif_path.unlink()
        out_bin.rename(tif_path)
        tifs = [tif_path]

    if not tifs:
        raise RuntimeError("No GeoTIFFs extracted from EE download.")
    return tifs


def build_rgb_from_tifs(tifs: list[Path]) -> Image.Image:
    def scale_u8(x: np.ndarray) -> np.ndarray:
        x = (x - MINN) / max(1.0, (MAXX - MINN))
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    if len(tifs) == 1:
        with rasterio.open(tifs[0]) as src:
            arr = src.read()  # (bands, H, W)
        if arr.shape[0] < 3:
            raise RuntimeError(f"Expected >=3 bands, got {arr.shape[0]}")

        r, g, b = arr[0].astype(np.float32), arr[1].astype(np.float32), arr[2].astype(np.float32)
        if max(r.max(), g.max(), b.max()) > 255:
            r8, g8, b8 = scale_u8(r), scale_u8(g), scale_u8(b)
        else:
            r8, g8, b8 = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

        return Image.merge("RGB", (
            Image.fromarray(r8, mode="L"),
            Image.fromarray(g8, mode="L"),
            Image.fromarray(b8, mode="L"),
        ))

    def find_band(tag: str) -> Path:
        for p in tifs:
            if tag in p.name:
                return p
        raise FileNotFoundError(f"Could not find {tag} among {[p.name for p in tifs]}")

    b4 = find_band("B4")
    b3 = find_band("B3")
    b2 = find_band("B2")

    def read_band_u8(p: Path) -> Image.Image:
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
        if a.max() > 255:
            a8 = scale_u8(a)
        else:
            a8 = a.astype(np.uint8)
        return Image.fromarray(a8, mode="L")

    return Image.merge("RGB", (read_band_u8(b4), read_band_u8(b3), read_band_u8(b2)))


def main() -> None:
    args = parse_args(sys.argv)
    input_dir: Path = args["input_dir"]
    output_dir: Path = args["output_dir"]
    start: str = args["start"]
    end: str = args["end"]
    maxcloud: float = args["maxcloud"]
    pattern: str = args["pattern"]
    skip_existing: bool = args["skip_existing"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize EE once
    ee.Initialize()

    specific_file = args["specific_file"]

    if specific_file is not None:
        tile_path = input_dir / specific_file
        if not tile_path.exists():
            raise FileNotFoundError(f"{tile_path} not found")
        tiles = [tile_path]
    else:
        tiles = sorted(input_dir.glob(pattern))

    if not tiles:
        raise RuntimeError(f"No tiles found in {input_dir}")

    print(f"[INFO] Found {len(tiles)} tile(s)")

    ok = 0
    skipped = 0
    failed = 0

    for tile_path in tiles:
        out_jpg = output_dir / (tile_path.stem + ".jpg")
        out_rgb_tif = output_dir / (tile_path.stem + "_rgb.tif")

        if skip_existing and out_jpg.exists() and out_rgb_tif.exists():
            skipped += 1
            continue

        tempdir = Path("tempdir")
        tempdir.mkdir(exist_ok=True)

        try:
            region_4326 = bounds_polygon_4326(tile_path)
            out_bin = download_s2_rgb(region_4326, tempdir, start=start, end=end, maxcloud=maxcloud)
            tifs = extract_tifs(out_bin, tempdir)
            rgb = build_rgb_from_tifs(tifs)

            # Resize RGB to match DSM tile pixel dimensions
            with Image.open(tile_path) as dem_img:
                w, h = dem_img.size
            rgb = rgb.resize((int(w), int(h)), Image.Resampling.LANCZOS)

            rgb.save(out_jpg, quality=95)
            rgb.save(out_rgb_tif)

            ok += 1
            if ok % 10 == 0:
                print(f"[INFO] Completed {ok} tiles...")

        except Exception as e:
            failed += 1
            print(f"[ERROR] {tile_path.name}: {e}")

        finally:
            # cleanup per-tile temp
            for p in tempdir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass

    print(f"\n[DONE] ok={ok}, skipped={skipped}, failed={failed}")
    print(f"[DONE] output_dir={output_dir}")


if __name__ == "__main__":
    main()