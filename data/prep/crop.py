"""
Tile GeoTIFF rasters into fixed-size square crops at a requested ground sample distance.

Supports:
  - DSM-only tiling (legacy/default workflow)
  - RGB-only tiling
  - Paired RGB + DSM tiling on a shared grid
  - Independent tiling for both RGB and DSM in one invocation

A tile is defined by:
  - tile_px x tile_px pixels
  - gsd meters / pixel

Examples:
  python crop.py
  python crop.py --tile_px 1024 --gsd 0.5
  python crop.py --in_dsm ../data/dsm/input --out_dsm ../data/dsm/output
  python crop.py --in_rgb ../data/rgb/input --out_rgb ../data/rgb/output
  python crop.py --in_rgb ../data/rgb/input --out_rgb ../data/rgb/output \
                  --in_dsm ../data/dsm/input --out_dsm ../data/dsm/output
  python crop.py --mode independent --in_rgb ../data/rgb/input --out_rgb ../data/rgb/output \
                  --in_dsm ../data/dsm/input --out_dsm ../data/dsm/output
  python crop.py my_raster.tif --single --tile_px 1024 --gsd 0.5

Notes:
  - Requires GDAL CLI: gdalwarp
  - Paired mode uses the strict intersection of RGB and DSM coverage.
  - Paired mode only writes tiles fully contained inside that overlap.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import rasterio
import rasterio.warp


DEFAULT_DSM_INPUT_DIR = Path("../data/dsm/input")
DEFAULT_DSM_OUTPUT_DIR = Path("../data/dsm/output")
VALID_SUFFIXES = {".tif", ".tiff"}
EPSILON = 1e-9
DTYPE_TO_GDAL = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    stem: str
    bounds: Bounds
    center_lon: float
    center_lat: float
    gdal_dtype: str | None
    nodata: float | int | str | None


@dataclass(frozen=True)
class ModalityJob:
    name: str
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class PairJob:
    key: str
    rgb_path: Path
    dsm_path: Path


@dataclass(frozen=True)
class TileConfig:
    tile_px: int
    overlap_px: int
    gsd: float
    single_mode: bool

    @property
    def tile_size_m(self) -> float:
        return self.tile_px * self.gsd

    @property
    def step_px(self) -> int:
        return self.tile_px - self.overlap_px

    @property
    def step_m(self) -> float:
        return self.step_px * self.gsd


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n  {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}"
        )


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def snap_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def snap_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile DSM and/or RGB GeoTIFF rasters into fixed-size metric tiles.",
    )
    parser.add_argument(
        "specific_file",
        nargs="?",
        help=(
            "Legacy positional filename for DSM-only workflows. The file is resolved "
            "relative to --in_dsm / --input_dir."
        ),
    )
    parser.add_argument(
        "--in_dsm",
        "--input_dsm_dir",
        "--input_dir",
        dest="in_dsm",
        type=Path,
        help="DSM input directory.",
    )
    parser.add_argument(
        "--out_dsm",
        "--output_dsm_dir",
        "--output_dir",
        dest="out_dsm",
        type=Path,
        help="DSM output directory.",
    )
    parser.add_argument(
        "--in_rgb",
        "--input_rgb_dir",
        dest="in_rgb",
        type=Path,
        help="RGB input directory.",
    )
    parser.add_argument(
        "--out_rgb",
        "--output_rgb_dir",
        dest="out_rgb",
        type=Path,
        help="RGB output directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "independent", "paired"],
        default="auto",
        help=(
            "auto: paired when both RGB and DSM are supplied, otherwise independent; "
            "independent: tile each supplied modality separately; "
            "paired: tile RGB and DSM together on a shared overlap grid."
        ),
    )
    parser.add_argument("--tile_px", type=int, default=1024, help="Tile width/height in pixels.")
    parser.add_argument(
        "--overlap_px",
        type=int,
        default=0,
        help="Overlap between adjacent tiles in pixels.",
    )
    parser.add_argument("--gsd", type=float, default=1.0, help="Meters per pixel.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Write a single centered tile per raster or per RGB/DSM pair.",
    )

    args = parser.parse_args()

    if args.tile_px <= 0:
        parser.error("--tile_px must be > 0")
    if args.gsd <= 0:
        parser.error("--gsd must be > 0")
    if args.overlap_px < 0:
        parser.error("--overlap_px must be >= 0")
    if args.overlap_px >= args.tile_px:
        parser.error("--overlap_px must be smaller than --tile_px")

    if args.in_dsm is None and args.in_rgb is None:
        args.in_dsm = DEFAULT_DSM_INPUT_DIR

    if args.out_dsm is None and args.in_dsm is not None and args.in_rgb is None:
        args.out_dsm = DEFAULT_DSM_OUTPUT_DIR

    if args.in_rgb is not None and args.out_rgb is None:
        parser.error("--out_rgb / --output_rgb_dir is required when --in_rgb is provided")
    if args.in_dsm is not None and args.out_dsm is None:
        parser.error("--out_dsm / --output_dsm_dir is required when --in_dsm is provided")

    if args.mode == "paired":
        if args.in_rgb is None or args.in_dsm is None:
            parser.error("--mode paired requires both RGB and DSM inputs and outputs")
        if args.out_rgb is None or args.out_dsm is None:
            parser.error("--mode paired requires both RGB and DSM outputs")

    if args.specific_file is not None:
        if args.in_dsm is None:
            parser.error("The positional filename is only supported for DSM input")
        if args.in_rgb is not None or args.mode == "paired":
            parser.error(
                "The positional filename is only supported for the legacy DSM-only workflow"
            )

    return args


def discover_rasters(input_dir: Path, specific_file: str | None = None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if specific_file is not None:
        tif_path = (input_dir / specific_file).resolve()
        if not tif_path.exists():
            raise FileNotFoundError(f"{tif_path} not found")
        if tif_path.suffix.lower() not in VALID_SUFFIXES:
            raise RuntimeError(f"File must be a GeoTIFF: {tif_path}")
        return [tif_path]

    rasters = sorted(
        p.resolve() for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if not rasters:
        raise RuntimeError(f"No .tif/.tiff files found in {input_dir}")
    return rasters


def get_dataset_info(tif_path: Path) -> DatasetInfo:
    with rasterio.open(tif_path) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Dataset has no CRS: {tif_path}")

        bounds = Bounds(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
        cx = bounds.center_x
        cy = bounds.center_y
        lon, lat = rasterio.warp.transform(ds.crs, "EPSG:4326", [cx], [cy])
        dtype_name = ds.dtypes[0] if ds.dtypes else None
        nodata = ds.nodata

    return DatasetInfo(
        path=tif_path,
        stem=tif_path.stem,
        bounds=bounds,
        center_lon=lon[0],
        center_lat=lat[0],
        gdal_dtype=DTYPE_TO_GDAL.get(dtype_name) if dtype_name is not None else None,
        nodata=nodata,
    )


def compute_common_utm(paths: Iterable[Path]) -> str:
    infos = [get_dataset_info(path) for path in paths]
    avg_lon = sum(info.center_lon for info in infos) / len(infos)
    avg_lat = sum(info.center_lat for info in infos) / len(infos)
    return f"EPSG:{utm_epsg_from_lonlat(avg_lon, avg_lat)}"


def dataset_bounds_in_crs(tif_path: Path, target_crs: str) -> Bounds:
    with rasterio.open(tif_path) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Dataset has no CRS: {tif_path}")
        bounds = ds.bounds
        crs = ds.crs

    xs = [bounds.left, bounds.right, bounds.right, bounds.left]
    ys = [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
    xt, yt = rasterio.warp.transform(crs, target_crs, xs, ys)
    return Bounds(min(xt), min(yt), max(xt), max(yt))


def intersect_bounds(a: Bounds, b: Bounds) -> Bounds | None:
    xmin = max(a.xmin, b.xmin)
    ymin = max(a.ymin, b.ymin)
    xmax = min(a.xmax, b.xmax)
    ymax = min(a.ymax, b.ymax)
    if xmax <= xmin + EPSILON or ymax <= ymin + EPSILON:
        return None
    return Bounds(xmin, ymin, xmax, ymax)


def tile_intersects_bounds(tile_bounds: Bounds, dataset_bounds: Bounds) -> bool:
    return not (
        tile_bounds.xmax <= dataset_bounds.xmin + EPSILON
        or tile_bounds.xmin >= dataset_bounds.xmax - EPSILON
        or tile_bounds.ymax <= dataset_bounds.ymin + EPSILON
        or tile_bounds.ymin >= dataset_bounds.ymax - EPSILON
    )


def iter_cover_starts(min_value: float, max_value: float, tile_size: float, step: float, gsd: float) -> list[float]:
    starts: list[float] = []
    current = snap_down(min_value, gsd)
    while current < max_value - EPSILON:
        if current + tile_size > min_value + EPSILON:
            starts.append(current)
        current += step
    return starts


def iter_contained_starts(
    min_value: float,
    max_value: float,
    tile_size: float,
    step: float,
    gsd: float,
) -> list[float]:
    starts: list[float] = []
    current = snap_up(min_value, gsd)
    while current + tile_size <= max_value + EPSILON:
        starts.append(current)
        current += step
    return starts


def build_tile_grid(bounds: Bounds, config: TileConfig, full_containment: bool) -> tuple[list[float], list[float]]:
    if full_containment:
        xs = iter_contained_starts(bounds.xmin, bounds.xmax, config.tile_size_m, config.step_m, config.gsd)
        ys = iter_contained_starts(bounds.ymin, bounds.ymax, config.tile_size_m, config.step_m, config.gsd)
    else:
        xs = iter_cover_starts(bounds.xmin, bounds.xmax, config.tile_size_m, config.step_m, config.gsd)
        ys = iter_cover_starts(bounds.ymin, bounds.ymax, config.tile_size_m, config.step_m, config.gsd)
    return xs, ys


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def single_tile_bounds(bounds: Bounds, config: TileConfig, full_containment: bool) -> Bounds:
    tile_size = config.tile_size_m

    if full_containment:
        min_x0 = snap_up(bounds.xmin, config.gsd)
        max_x0 = snap_down(bounds.xmax - tile_size, config.gsd)
        min_y0 = snap_up(bounds.ymin, config.gsd)
        max_y0 = snap_down(bounds.ymax - tile_size, config.gsd)

        if min_x0 > max_x0 + EPSILON or min_y0 > max_y0 + EPSILON:
            raise RuntimeError(
                "Overlap extent is smaller than one fully-contained tile at the requested tile size / gsd"
            )

        x0 = snap_down(bounds.center_x - tile_size / 2.0, config.gsd)
        y0 = snap_down(bounds.center_y - tile_size / 2.0, config.gsd)
        x0 = clamp(x0, min_x0, max_x0)
        y0 = clamp(y0, min_y0, max_y0)
    else:
        x0 = snap_down(bounds.center_x - tile_size / 2.0, config.gsd)
        y0 = snap_down(bounds.center_y - tile_size / 2.0, config.gsd)

    return Bounds(x0, y0, x0 + tile_size, y0 + tile_size)


def format_nodata(value: float | int | str | None) -> str | None:
    if value is None:
        return None
    return str(value)


def build_gdalwarp_command(
    in_path: Path,
    out_path: Path,
    target_crs: str,
    tile_bounds: Bounds,
    gsd: float,
    gdal_dtype: str | None,
    nodata: float | int | str | None,
) -> list[str]:
    cmd = [
        "gdalwarp",
        "-t_srs",
        target_crs,
        "-te",
        f"{tile_bounds.xmin:.12f}",
        f"{tile_bounds.ymin:.12f}",
        f"{tile_bounds.xmax:.12f}",
        f"{tile_bounds.ymax:.12f}",
        "-tr",
        f"{gsd:.12f}",
        f"{gsd:.12f}",
        "-r",
        "bilinear",
        "-overwrite",
    ]

    if gdal_dtype is not None:
        cmd.extend(["-ot", gdal_dtype])

    nodata_str = format_nodata(nodata)
    if nodata_str is not None:
        cmd.extend(["-srcnodata", nodata_str, "-dstnodata", nodata_str])

    cmd.extend([str(in_path), str(out_path)])
    return cmd


def write_tile(
    in_path: Path,
    out_path: Path,
    target_crs: str,
    tile_bounds: Bounds,
    gsd: float,
    gdal_dtype: str | None,
    nodata: float | int | str | None,
) -> None:
    run(build_gdalwarp_command(in_path, out_path, target_crs, tile_bounds, gsd, gdal_dtype, nodata))


def print_grid_summary(label: str, bounds: Bounds, config: TileConfig, xs: list[float], ys: list[float], target_crs: str) -> None:
    print(f"\n[INFO] {label}")
    print(f"[INFO] CRS: {target_crs}")
    print(f"[INFO] Tile size: {config.tile_px} px x {config.tile_px} px")
    print(f"[INFO] GSD: {config.gsd} m/px")
    print(
        f"[INFO] Overlap: {config.overlap_px} px "
        f"(step={config.step_px} px / {config.step_m} m)"
    )
    print(
        f"[INFO] Bounds: xmin={format_metric(bounds.xmin)}, ymin={format_metric(bounds.ymin)}, "
        f"xmax={format_metric(bounds.xmax)}, ymax={format_metric(bounds.ymax)}"
    )
    print(
        f"[INFO] Ground tile size: {format_metric(config.tile_size_m)} m x "
        f"{format_metric(config.tile_size_m)} m"
    )
    print(f"[INFO] Grid: {len(ys)} rows x {len(xs)} cols")


def print_single_summary(label: str, crop_bounds: Bounds, config: TileConfig, target_crs: str) -> None:
    print(f"\n[INFO] {label}")
    print(f"[INFO] CRS: {target_crs}")
    print(f"[INFO] Tile size: {config.tile_px} px x {config.tile_px} px")
    print(f"[INFO] GSD: {config.gsd} m/px")
    print(
        f"[INFO] Crop bounds: xmin={format_metric(crop_bounds.xmin)}, "
        f"ymin={format_metric(crop_bounds.ymin)}, xmax={format_metric(crop_bounds.xmax)}, "
        f"ymax={format_metric(crop_bounds.ymax)}"
    )


def output_name(base_key: str, row: int, col: int, single_mode: bool, config: TileConfig) -> str:
    if single_mode:
        gsd_text = str(config.gsd).replace(".", "p")
        return f"{base_key}_single_{config.tile_px}px_{gsd_text}mpp.tif"
    return f"{base_key}_tile_{row:04d}_{col:04d}.tif"


def tile_independent_raster(in_path: Path, out_dir: Path, config: TileConfig) -> int:
    info = get_dataset_info(in_path)
    target_crs = compute_common_utm([in_path])
    bounds = dataset_bounds_in_crs(in_path, target_crs)

    out_dir.mkdir(parents=True, exist_ok=True)

    if config.single_mode:
        crop_bounds = single_tile_bounds(bounds, config, full_containment=False)
        out_path = out_dir / output_name(info.stem, 0, 0, True, config)
        print_single_summary(f"Cropping SINGLE tile from: {in_path.name}", crop_bounds, config, target_crs)
        write_tile(
            in_path,
            out_path,
            target_crs,
            crop_bounds,
            config.gsd,
            info.gdal_dtype,
            info.nodata,
        )
        print(f"[DONE] Wrote single tile: {out_path}")
        return 1

    xs, ys = build_tile_grid(bounds, config, full_containment=False)
    print_grid_summary(f"Tiling: {in_path.name}", bounds, config, xs, ys, target_crs)

    written = 0
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            tile_bounds = Bounds(x0, y0, x0 + config.tile_size_m, y0 + config.tile_size_m)
            if not tile_intersects_bounds(tile_bounds, bounds):
                continue

            out_path = out_dir / output_name(info.stem, row, col, False, config)
            write_tile(
                in_path,
                out_path,
                target_crs,
                tile_bounds,
                config.gsd,
                info.gdal_dtype,
                info.nodata,
            )
            written += 1

    print(f"[DONE] {in_path.name}: wrote {written} tile(s)")
    return written


def build_unique_stem_map(paths: list[Path], label: str) -> dict[str, Path]:
    stem_map: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        if path.stem in stem_map:
            duplicates.append(path.stem)
        else:
            stem_map[path.stem] = path
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates)))
        raise RuntimeError(f"Duplicate {label} stems found: {duplicate_list}")
    return stem_map


def match_rgb_dsm_pairs(rgb_paths: list[Path], dsm_paths: list[Path]) -> list[PairJob]:
    rgb_map = build_unique_stem_map(rgb_paths, "RGB")
    dsm_map = build_unique_stem_map(dsm_paths, "DSM")

    if len(rgb_paths) == 1 and len(dsm_paths) == 1:
        rgb_path = rgb_paths[0]
        dsm_path = dsm_paths[0]
        if rgb_path.stem == dsm_path.stem:
            key = rgb_path.stem
        else:
            key = f"{rgb_path.stem}__{dsm_path.stem}"
        return [PairJob(key=key, rgb_path=rgb_path, dsm_path=dsm_path)]

    rgb_stems = set(rgb_map)
    dsm_stems = set(dsm_map)
    missing_rgb = sorted(dsm_stems - rgb_stems)
    missing_dsm = sorted(rgb_stems - dsm_stems)

    if missing_rgb or missing_dsm:
        problems: list[str] = []
        if missing_dsm:
            problems.append("missing DSM for RGB stems: " + ", ".join(missing_dsm))
        if missing_rgb:
            problems.append("missing RGB for DSM stems: " + ", ".join(missing_rgb))
        raise RuntimeError("Paired mode requires exact RGB/DSM pairing by stem; " + "; ".join(problems))

    pairs = [
        PairJob(key=stem, rgb_path=rgb_map[stem], dsm_path=dsm_map[stem])
        for stem in sorted(rgb_stems & dsm_stems)
    ]
    if not pairs:
        raise RuntimeError("No RGB/DSM pairs found")
    return pairs


def tile_paired_rasters(
    pair: PairJob,
    rgb_out_dir: Path,
    dsm_out_dir: Path,
    config: TileConfig,
) -> int:
    rgb_info = get_dataset_info(pair.rgb_path)
    dsm_info = get_dataset_info(pair.dsm_path)
    target_crs = compute_common_utm([pair.rgb_path, pair.dsm_path])

    rgb_bounds = dataset_bounds_in_crs(pair.rgb_path, target_crs)
    dsm_bounds = dataset_bounds_in_crs(pair.dsm_path, target_crs)
    overlap_bounds = intersect_bounds(rgb_bounds, dsm_bounds)
    if overlap_bounds is None:
        raise RuntimeError(
            f"RGB/DSM pair has no overlap in {target_crs}: {pair.rgb_path.name} vs {pair.dsm_path.name}"
        )

    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    dsm_out_dir.mkdir(parents=True, exist_ok=True)

    if config.single_mode:
        crop_bounds = single_tile_bounds(overlap_bounds, config, full_containment=True)
        rgb_out = rgb_out_dir / output_name(pair.key, 0, 0, True, config)
        dsm_out = dsm_out_dir / output_name(pair.key, 0, 0, True, config)

        print_single_summary(
            f"Cropping SINGLE paired tile from RGB={pair.rgb_path.name}, DSM={pair.dsm_path.name}",
            crop_bounds,
            config,
            target_crs,
        )
        write_tile(
            pair.rgb_path,
            rgb_out,
            target_crs,
            crop_bounds,
            config.gsd,
            rgb_info.gdal_dtype,
            rgb_info.nodata,
        )
        write_tile(
            pair.dsm_path,
            dsm_out,
            target_crs,
            crop_bounds,
            config.gsd,
            dsm_info.gdal_dtype,
            dsm_info.nodata,
        )
        print(f"[DONE] {pair.key}: wrote paired single tile")
        return 1

    xs, ys = build_tile_grid(overlap_bounds, config, full_containment=True)
    print_grid_summary(
        f"Paired tiling: RGB={pair.rgb_path.name}, DSM={pair.dsm_path.name}",
        overlap_bounds,
        config,
        xs,
        ys,
        target_crs,
    )

    if not xs or not ys:
        raise RuntimeError(
            f"No fully-contained paired tiles fit within the RGB/DSM overlap for pair {pair.key}"
        )

    written = 0
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            tile_bounds = Bounds(x0, y0, x0 + config.tile_size_m, y0 + config.tile_size_m)
            rgb_out = rgb_out_dir / output_name(pair.key, row, col, False, config)
            dsm_out = dsm_out_dir / output_name(pair.key, row, col, False, config)

            write_tile(
                pair.rgb_path,
                rgb_out,
                target_crs,
                tile_bounds,
                config.gsd,
                rgb_info.gdal_dtype,
                rgb_info.nodata,
            )
            write_tile(
                pair.dsm_path,
                dsm_out,
                target_crs,
                tile_bounds,
                config.gsd,
                dsm_info.gdal_dtype,
                dsm_info.nodata,
            )
            written += 1

    print(f"[DONE] {pair.key}: wrote {written} paired tile(s)")
    return written


def determine_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if args.in_rgb is not None and args.in_dsm is not None:
        return "paired"
    return "independent"


def build_independent_jobs(args: argparse.Namespace) -> list[ModalityJob]:
    jobs: list[ModalityJob] = []
    if args.in_rgb is not None:
        jobs.append(ModalityJob(name="rgb", input_dir=args.in_rgb.resolve(), output_dir=args.out_rgb.resolve()))
    if args.in_dsm is not None:
        jobs.append(ModalityJob(name="dsm", input_dir=args.in_dsm.resolve(), output_dir=args.out_dsm.resolve()))
    return jobs


def main() -> None:
    args = parse_args()
    config = TileConfig(
        tile_px=args.tile_px,
        overlap_px=args.overlap_px,
        gsd=args.gsd,
        single_mode=args.single,
    )

    mode = determine_mode(args)

    if mode == "independent":
        jobs = build_independent_jobs(args)
        if not jobs:
            raise RuntimeError("No RGB or DSM inputs were provided")

        total_written = 0
        for job in jobs:
            specific_file = args.specific_file if job.name == "dsm" else None
            rasters = discover_rasters(job.input_dir, specific_file=specific_file)
            modality_written = 0
            for tif_path in rasters:
                modality_written += tile_independent_raster(tif_path, job.output_dir, config)
            total_written += modality_written
            print(f"[ALL DONE] {job.name.upper()} output dir: {job.output_dir}")
            print(f"[ALL DONE] {job.name.upper()} tiles written: {modality_written}")

        print(f"\n[ALL DONE] Total tiles written: {total_written}")
        return

    rgb_paths = discover_rasters(args.in_rgb.resolve())
    dsm_paths = discover_rasters(args.in_dsm.resolve())
    pairs = match_rgb_dsm_pairs(rgb_paths, dsm_paths)

    total_written = 0
    rgb_out_dir = args.out_rgb.resolve()
    dsm_out_dir = args.out_dsm.resolve()
    for pair in pairs:
        total_written += tile_paired_rasters(pair, rgb_out_dir, dsm_out_dir, config)

    print(f"\n[ALL DONE] Paired tiles written: {total_written}")
    print(f"[ALL DONE] RGB output dir: {rgb_out_dir}")
    print(f"[ALL DONE] DSM output dir: {dsm_out_dir}")


if __name__ == "__main__":
    main()