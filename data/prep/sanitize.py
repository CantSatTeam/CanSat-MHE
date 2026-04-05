"""
Data quality auditor for monocular height estimation datasets.

Expects a dataset of paired (RGB image, height map) tiles.
Runs three checks in priority order:
  1. RGB–label spatial misalignment (edge correlation)
  2. Height distribution outliers (datum/offset errors)
  3. Void fraction in height labels

Outputs a CSV report and a list of rejected tile paths.

Usage:
    python clean_height_dataset.py \
        --rgb_dir   /data/rgb \
        --dsm_dir   /data/dsm \
        --out_dir   /data/clean \
        --report    audit_report.csv \
        --void_thresh     0.05 \
        --align_thresh    0.10 \
        --zscore_thresh   3.0

Assumptions:
  - RGB tiles are .tif / .png / .jpg
  - Height map tiles are .tif (float32, metres, nodata = NaN or a known fill value)
  - Filenames are paired by stem (e.g. tile_042.png <-> tile_042.tif)
  - Rasterio is used for height maps; PIL / OpenCV for RGB
"""

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not found — falling back to PIL for height maps. "
                    "Nodata handling may be less accurate.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TileResult:
    stem: str
    rgb_path: str
    dsm_path: str

    # Check 1 – alignment
    edge_correlation: float = float("nan")
    alignment_pass: Optional[bool] = None

    # Check 2 – height distribution
    height_median: float = float("nan")
    height_range: float = float("nan")
    height_zscore: float = float("nan")
    distribution_pass: Optional[bool] = None

    # Check 3 – void fraction
    void_fraction: float = float("nan")
    void_pass: Optional[bool] = None

    # Aggregate
    passed: bool = False
    reject_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

RGB_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DSM_EXTS = {".tif", ".tiff"}


def _load_rgb_gray(path: Path) -> np.ndarray:
    """Load RGB tile, return float32 grayscale [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    # Luminosity weights
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return gray


def _load_dsm(path: Path, nodata_fill: float = np.nan) -> np.ndarray:
    """Load height map as float32 array; NaN where nodata."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None:
                data[data == nd] = np.nan
    else:
        img = Image.open(path)
        data = np.array(img, dtype=np.float32)
    return data


def _find_pairs(rgb_dir: Path, dsm_dir: Path):
    """Match RGB and DSM tiles by filename stem."""
    rgb_files = {p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in RGB_EXTS}
    dsm_files = {p.stem: p for p in dsm_dir.iterdir() if p.suffix.lower() in DSM_EXTS}
    stems = sorted(set(rgb_files) & set(dsm_files))
    missing_dsm = set(rgb_files) - set(dsm_files)
    missing_rgb = set(dsm_files) - set(rgb_files)
    if missing_dsm:
        log.warning("%d RGB tiles have no matching DSM: %s …",
                    len(missing_dsm), list(missing_dsm)[:5])
    if missing_rgb:
        log.warning("%d DSM tiles have no matching RGB: %s …",
                    len(missing_rgb), list(missing_rgb)[:5])
    return [(s, rgb_files[s], dsm_files[s]) for s in stems]


# ---------------------------------------------------------------------------
# Check 1 – RGB / height spatial alignment
# ---------------------------------------------------------------------------

def _sobel_edges(arr: np.ndarray) -> np.ndarray:
    """Return normalised Sobel edge magnitude."""
    kx = ndimage.sobel(arr, axis=1)
    ky = ndimage.sobel(arr, axis=0)
    mag = np.hypot(kx, ky)
    mx = mag.max()
    return mag / mx if mx > 0 else mag


def check_alignment(rgb_gray: np.ndarray, dsm: np.ndarray,
                    thresh: float = 0.10) -> tuple[float, bool]:
    """
    Pearson correlation between Sobel edge maps of the RGB and DSM.

    In structured terrain (buildings, roads) edges in colour and height
    should coincide.  Very low correlation implies misalignment.

    Returns (correlation, passed).
    """
    # Resize DSM to match RGB if needed
    if rgb_gray.shape != dsm.shape:
        from PIL import Image as _Image
        dsm_img = _Image.fromarray(dsm).resize(
            (rgb_gray.shape[1], rgb_gray.shape[0]), _Image.BILINEAR)
        dsm_resized = np.array(dsm_img, dtype=np.float32)
    else:
        dsm_resized = dsm.copy()

    # Replace NaN in dsm with interpolated values for edge detection
    mask = np.isnan(dsm_resized)
    if mask.any():
        dsm_resized[mask] = np.nanmedian(dsm_resized)

    edges_rgb = _sobel_edges(rgb_gray)
    edges_dsm = _sobel_edges(dsm_resized)

    # Pearson correlation of edge maps
    flat_r = edges_rgb.ravel()
    flat_d = edges_dsm.ravel()
    corr = float(np.corrcoef(flat_r, flat_d)[0, 1])

    return corr, corr >= thresh


# ---------------------------------------------------------------------------
# Check 2 – Height distribution outliers
# ---------------------------------------------------------------------------

def compute_tile_height_stats(dsm: np.ndarray) -> tuple[float, float]:
    """Return (median, height_range) ignoring NaN."""
    valid = dsm[~np.isnan(dsm)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(np.median(valid)), float(valid.max() - valid.min())


def check_distribution(median: float, all_medians: np.ndarray,
                        height_range: float,
                        zscore_thresh: float = 3.0,
                        min_range: float = 0.0,
                        max_range: float = 3000.0) -> tuple[float, bool]:
    """
    Flag tiles whose median height is a z-score outlier across the dataset,
    or whose height range is physically implausible.

    Returns (zscore, passed).
    """
    if np.isnan(median):
        return float("nan"), False

    pop_median = np.nanmedian(all_medians)
    pop_std = np.nanstd(all_medians)
    zscore = abs(median - pop_median) / (pop_std + 1e-6)

    range_ok = min_range <= height_range <= max_range
    passed = zscore <= zscore_thresh and range_ok
    return float(zscore), passed


# ---------------------------------------------------------------------------
# Check 3 – Void fraction
# ---------------------------------------------------------------------------

def check_void_fraction(dsm: np.ndarray,
                         thresh: float = 0.05) -> tuple[float, bool]:
    """
    Fraction of NaN / no-data pixels in the height map.

    Returns (void_fraction, passed).
    """
    total = dsm.size
    if total == 0:
        return 1.0, False
    n_nan = int(np.isnan(dsm).sum())
    frac = n_nan / total
    return frac, frac <= thresh


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------

def audit_dataset(rgb_dir: Path,
                  dsm_dir: Path,
                  out_dir: Optional[Path],
                  report_path: Path,
                  void_thresh: float = 0.05,
                  align_thresh: float = 0.10,
                  zscore_thresh: float = 3.0,
                  copy_passing: bool = False) -> list[TileResult]:

    pairs = _find_pairs(rgb_dir, dsm_dir)
    log.info("Found %d paired tiles.", len(pairs))

    results: list[TileResult] = []

    # ---- Pass 1: load stats needed for dataset-wide z-score ----------------
    log.info("Pass 1/2 — computing per-tile height stats …")
    tile_medians: list[float] = []
    raw_stats: list[tuple] = []  # (stem, rgb_path, dsm_path, dsm_array)

    for stem, rgb_path, dsm_path in pairs:
        dsm = _load_dsm(dsm_path)
        med, rng = compute_tile_height_stats(dsm)
        tile_medians.append(med)
        raw_stats.append((stem, rgb_path, dsm_path, dsm, med, rng))

    all_medians = np.array(tile_medians, dtype=np.float64)

    # ---- Pass 2: run all checks --------------------------------------------
    log.info("Pass 2/2 — running quality checks …")

    if out_dir:
        (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (out_dir / "dsm").mkdir(parents=True, exist_ok=True)
        (out_dir / "rejected").mkdir(parents=True, exist_ok=True)

    for stem, rgb_path, dsm_path, dsm, med, rng in raw_stats:
        res = TileResult(stem=stem,
                         rgb_path=str(rgb_path),
                         dsm_path=str(dsm_path))

        # -- Check 3: void fraction (cheapest, run first) --------------------
        res.void_fraction, res.void_pass = check_void_fraction(dsm, void_thresh)
        if not res.void_pass:
            res.reject_reasons.append(
                f"void_fraction={res.void_fraction:.3f} > {void_thresh}")

        # -- Check 2: distribution outlier -----------------------------------
        res.height_median = med
        res.height_range = rng
        res.height_zscore, res.distribution_pass = check_distribution(
            med, all_medians, rng, zscore_thresh)
        if not res.distribution_pass:
            res.reject_reasons.append(
                f"height_zscore={res.height_zscore:.2f} > {zscore_thresh} "
                f"or range={rng:.1f}m implausible")

        # -- Check 1: alignment (most expensive, run last) -------------------
        try:
            rgb_gray = _load_rgb_gray(rgb_path)
            res.edge_correlation, res.alignment_pass = check_alignment(
                rgb_gray, dsm, align_thresh)
        except Exception as e:
            log.warning("Alignment check failed for %s: %s", stem, e)
            res.alignment_pass = False
            res.reject_reasons.append(f"alignment_check_error: {e}")

        if not res.alignment_pass:
            res.reject_reasons.append(
                f"edge_correlation={res.edge_correlation:.3f} < {align_thresh}")

        res.passed = (res.void_pass and res.distribution_pass and res.alignment_pass)

        # -- Optionally copy files -------------------------------------------
        if out_dir and copy_passing:
            dest_sub = "rgb" if res.passed else "rejected"
            if res.passed:
                shutil.copy2(rgb_path, out_dir / "rgb" / rgb_path.name)
                shutil.copy2(dsm_path, out_dir / "dsm" / dsm_path.name)
            else:
                shutil.copy2(rgb_path, out_dir / "rejected" / rgb_path.name)

        status = "PASS" if res.passed else f"FAIL [{'; '.join(res.reject_reasons)}]"
        log.info("  %-30s %s", stem, status)
        results.append(res)

    # ---- Write report -------------------------------------------------------
    _write_report(results, report_path)

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    log.info("Done. %d/%d tiles passed (%d rejected).", n_pass, len(results), n_fail)
    _print_summary(results, void_thresh, align_thresh, zscore_thresh)

    return results


def _write_report(results: list[TileResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stem", "passed", "reject_reasons",
        "edge_correlation", "alignment_pass",
        "height_median", "height_range", "height_zscore", "distribution_pass",
        "void_fraction", "void_pass",
        "rgb_path", "dsm_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["reject_reasons"] = " | ".join(row["reject_reasons"])
            writer.writerow({k: row[k] for k in fields})
    log.info("Report written to %s", path)


def _print_summary(results, void_thresh, align_thresh, zscore_thresh):
    n = len(results)
    if n == 0:
        return
    n_void  = sum(not r.void_pass         for r in results if r.void_pass         is not None)
    n_dist  = sum(not r.distribution_pass for r in results if r.distribution_pass is not None)
    n_align = sum(not r.alignment_pass    for r in results if r.alignment_pass    is not None)

    print("\n" + "="*55)
    print(f"  AUDIT SUMMARY  ({n} tiles)")
    print("="*55)
    print(f"  Check 1 – Alignment      (thresh ≥{align_thresh:.2f}):  "
          f"{n_align:4d} rejected  ({100*n_align/n:.1f}%)")
    print(f"  Check 2 – Distribution   (z  ≤{zscore_thresh:.1f}):      "
          f"{n_dist:4d} rejected  ({100*n_dist/n:.1f}%)")
    print(f"  Check 3 – Void fraction  (≤{void_thresh:.2f}):       "
          f"{n_void:4d} rejected  ({100*n_void/n:.1f}%)")
    print("-"*55)
    n_pass = sum(r.passed for r in results)
    print(f"  Total passing:  {n_pass}/{n}  ({100*n_pass/n:.1f}%)")
    print("="*55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Height dataset quality auditor")
    p.add_argument("--rgb_dir",        required=True,  type=Path)
    p.add_argument("--dsm_dir",        required=True,  type=Path)
    p.add_argument("--out_dir",        default=None,   type=Path,
                   help="If set, copies passing tiles here (rgb/ and dsm/ subdirs)")
    p.add_argument("--report",         default="audit_report.csv", type=Path)
    p.add_argument("--void_thresh",    default=0.05,  type=float,
                   help="Max void/NaN fraction in DSM (default 0.05)")
    p.add_argument("--align_thresh",   default=0.10,  type=float,
                   help="Min edge correlation RGB↔DSM (default 0.10)")
    p.add_argument("--zscore_thresh",  default=3.0,   type=float,
                   help="Max z-score for tile median height (default 3.0)")
    p.add_argument("--copy",           action="store_true",
                   help="Copy passing tiles to out_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit_dataset(
        rgb_dir=args.rgb_dir,
        dsm_dir=args.dsm_dir,
        out_dir=args.out_dir,
        report_path=args.report,
        void_thresh=args.void_thresh,
        align_thresh=args.align_thresh,
        zscore_thresh=args.zscore_thresh,
        copy_passing=args.copy,
    )