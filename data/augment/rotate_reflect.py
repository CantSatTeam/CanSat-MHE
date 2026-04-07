"""
Geometric augmentation for paired (RGB, height map) tiles.

Applies random rotations and reflections identically to both the RGB image
and its paired height map, preserving pixel-level correspondence.

Augmentations:
  - Horizontal flip
  - Vertical flip
  - 90 / 180 / 270 degree rotations
  - Combinations of the above (full dihedral group D4 — 8 total transforms)

Height values are preserved exactly (no interpolation for 90-degree multiples).
For arbitrary-angle rotation, bilinear interpolation is used with NaN-fill at
borders.

Usage:
    # Deterministic: generate all 8 D4 transforms for every tile
    python augment_tiles.py \
        --rgb_dir /data/clean/rgb \
        --dsm_dir /data/clean/dsm \
        --out_dir /data/augmented \
        --mode d4

    # Random: sample N random transforms per tile
    python augment_tiles.py \
        --rgb_dir /data/clean/rgb \
        --dsm_dir /data/clean/dsm \
        --out_dir /data/augmented \
        --mode random \
        --n_per_tile 3 \
        --seed 42

    # Arbitrary angles (adds random rotation in [min_angle, max_angle])
    python augment_tiles.py \
        --rgb_dir /data/clean/rgb \
        --dsm_dir /data/clean/dsm \
        --out_dir /data/augmented \
        --mode random \
        --n_per_tile 4 \
        --arbitrary_angles \
        --min_angle -30 \
        --max_angle  30
"""

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.ndimage import rotate as scipy_rotate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RGB_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
DSM_EXTS = {".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Transform definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    """A member of the dihedral group D4: flip + rotation."""
    flip_h: bool   # horizontal flip (left-right)
    flip_v: bool   # vertical flip   (up-down)
    rot90: int     # number of 90-degree CCW rotations (0-3)
    angle: float = 0.0  # additional arbitrary rotation (degrees)

    @property
    def name(self) -> str:
        parts = []
        if self.flip_h: parts.append("fh")
        if self.flip_v: parts.append("fv")
        if self.rot90:  parts.append(f"r{self.rot90 * 90}")
        if self.angle:  parts.append(f"a{self.angle:.0f}")
        return "_".join(parts) if parts else "identity"


# All 8 elements of D4
D4_TRANSFORMS = [
    Transform(flip_h=False, flip_v=False, rot90=0),
    Transform(flip_h=False, flip_v=False, rot90=1),
    Transform(flip_h=False, flip_v=False, rot90=2),
    Transform(flip_h=False, flip_v=False, rot90=3),
    Transform(flip_h=True,  flip_v=False, rot90=0),
    Transform(flip_h=True,  flip_v=False, rot90=1),
    Transform(flip_h=True,  flip_v=False, rot90=2),
    Transform(flip_h=True,  flip_v=False, rot90=3),
]
# Note: flip_v + rot90 covers the remaining D4 elements via equivalence,
# but we list them explicitly for clarity rather than deduplication.


# ---------------------------------------------------------------------------
# Core transform logic
# ---------------------------------------------------------------------------

def _apply_to_array(arr: np.ndarray, t: Transform,
                    is_height: bool = False) -> np.ndarray:
    """
    Apply transform t to a 2D (H, W) or 3D (H, W, C) numpy array.
    Height maps stay float32; RGB stays uint8.
    """
    if t.flip_h:
        arr = np.fliplr(arr)
    if t.flip_v:
        arr = np.flipud(arr)
    if t.rot90:
        arr = np.rot90(arr, k=t.rot90)

    if t.angle != 0.0:
        if not HAS_SCIPY:
            raise RuntimeError(
                "scipy is required for arbitrary-angle rotation. "
                "Install with: pip install scipy")
        if is_height:
            # NaN-safe rotation: fill NaN with median, rotate, restore NaN mask
            nan_mask = np.isnan(arr)
            fill_val = float(np.nanmedian(arr)) if not nan_mask.all() else 0.0
            arr_filled = np.where(nan_mask, fill_val, arr)
            arr_rot = scipy_rotate(arr_filled, angle=t.angle,
                                   reshape=False, order=1,
                                   mode="constant", cval=fill_val)
            mask_rot = scipy_rotate(nan_mask.astype(np.float32), angle=t.angle,
                                    reshape=False, order=0,
                                    mode="constant", cval=1.0) > 0.5
            arr_rot[mask_rot] = np.nan
            arr = arr_rot.astype(np.float32)
        else:
            # RGB: rotate each channel, clamp to [0, 255]
            if arr.ndim == 3:
                channels = [
                    scipy_rotate(arr[..., c], angle=t.angle,
                                 reshape=False, order=1,
                                 mode="reflect")
                    for c in range(arr.shape[2])
                ]
                arr = np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)
            else:
                arr = scipy_rotate(arr, angle=t.angle,
                                   reshape=False, order=1,
                                   mode="reflect").clip(0, 255).astype(np.uint8)

    return arr


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_dsm(path: Path) -> tuple[np.ndarray, dict]:
    """Returns (array float32 with NaN for nodata, meta dict)."""
    meta = {}
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None:
                data[data == nd] = np.nan
            meta = dict(src.meta)
    else:
        data = np.array(Image.open(path), dtype=np.float32)
    return data, meta


def _save_rgb(arr: np.ndarray, path: Path):
    Image.fromarray(arr, mode="RGB").save(path)


def _save_dsm(arr: np.ndarray, path: Path, meta: dict):
    if HAS_RASTERIO and meta:
        out_meta = meta.copy()
        out_meta.update({
            "height": arr.shape[0],
            "width":  arr.shape[1],
            "count":  1,
            "dtype":  "float32",
            "nodata": np.nan,
        })
        with rasterio.open(path, "w", **out_meta) as dst:
            nan_mask = np.isnan(arr)
            out = arr.copy()
            out[nan_mask] = out_meta["nodata"]
            dst.write(out, 1)
    else:
        # Fallback: save as 32-bit TIFF via PIL
        img = Image.fromarray(arr, mode="F")
        img.save(path)


def _find_pairs(rgb_dir: Path, dsm_dir: Path):
    rgb_files = {p.stem: p for p in rgb_dir.iterdir()
                 if p.suffix.lower() in RGB_EXTS}
    dsm_files = {p.stem: p for p in dsm_dir.iterdir()
                 if p.suffix.lower() in DSM_EXTS}
    stems = sorted(set(rgb_files) & set(dsm_files))
    log.info("Found %d paired tiles.", len(stems))
    return [(s, rgb_files[s], dsm_files[s]) for s in stems]


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def _make_random_transform(rng: random.Random,
                            arbitrary_angles: bool,
                            min_angle: float,
                            max_angle: float) -> Transform:
    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5
    rot90  = rng.randint(0, 3)
    angle  = rng.uniform(min_angle, max_angle) if arbitrary_angles else 0.0
    return Transform(flip_h=flip_h, flip_v=flip_v, rot90=rot90, angle=angle)


def augment_dataset(rgb_dir: Path,
                    dsm_dir: Path,
                    out_dir: Path,
                    mode: str = "d4",
                    n_per_tile: int = 3,
                    include_identity: bool = False,
                    arbitrary_angles: bool = False,
                    min_angle: float = -30.0,
                    max_angle: float = 30.0,
                    seed: int = 0,
                    rgb_ext: str = ".png",
                    dsm_ext: str = ".tif"):

    rng = random.Random(seed)
    pairs = _find_pairs(rgb_dir, dsm_dir)

    out_rgb = out_dir / "rgb"
    out_dsm = out_dir / "dsm"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_dsm.mkdir(parents=True, exist_ok=True)

    total_written = 0

    for stem, rgb_path, dsm_path in pairs:
        rgb_arr = _load_rgb(rgb_path)
        dsm_arr, dsm_meta = _load_dsm(dsm_path)

        if mode == "d4":
            transforms = [t for t in D4_TRANSFORMS
                          if include_identity or t.name != "identity"]
        elif mode == "random":
            transforms = [
                _make_random_transform(rng, arbitrary_angles, min_angle, max_angle)
                for _ in range(n_per_tile)
            ]
            # Deduplicate by name (not possible for arbitrary angles)
            if not arbitrary_angles:
                seen = set()
                unique = []
                for t in transforms:
                    if t.name not in seen:
                        seen.add(t.name)
                        unique.append(t)
                transforms = unique
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'd4' or 'random'.")

        for t in transforms:
            aug_stem = f"{stem}__{t.name}"

            aug_rgb = _apply_to_array(rgb_arr.copy(), t, is_height=False)
            aug_dsm = _apply_to_array(dsm_arr.copy(), t, is_height=True)

            _save_rgb(aug_rgb, out_rgb / f"{aug_stem}{rgb_ext}")
            _save_dsm(aug_dsm, out_dsm / f"{aug_stem}{dsm_ext}", dsm_meta)

            total_written += 1

        log.info("  %s → %d augmented tiles", stem, len(transforms))

    log.info("Done. %d augmented tile pairs written to %s.", total_written, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Geometric augmentation for height dataset")
    p.add_argument("--rgb_dir",  required=True, type=Path)
    p.add_argument("--dsm_dir",  required=True, type=Path)
    p.add_argument("--out_dir",  required=True, type=Path)

    p.add_argument("--mode", default="d4", choices=["d4", "random"],
                   help="d4: all 8 dihedral transforms; random: sample N per tile")
    p.add_argument("--n_per_tile", default=3, type=int,
                   help="Number of random transforms per tile (mode=random only)")
    p.add_argument("--include_identity", action="store_true",
                   help="Include the original (unaugmented) tile in output")
    p.add_argument("--seed", default=42, type=int)

    p.add_argument("--arbitrary_angles", action="store_true",
                   help="Also apply a random rotation in [min_angle, max_angle] degrees")
    p.add_argument("--min_angle", default=-30.0, type=float)
    p.add_argument("--max_angle",  default=30.0, type=float)

    p.add_argument("--rgb_ext", default=".png",
                   help="Output extension for RGB tiles (default: .png)")
    p.add_argument("--dsm_ext", default=".tif",
                   help="Output extension for DSM tiles (default: .tif)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_dataset(
        rgb_dir=args.rgb_dir,
        dsm_dir=args.dsm_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        n_per_tile=args.n_per_tile,
        include_identity=args.include_identity,
        arbitrary_angles=args.arbitrary_angles,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        seed=args.seed,
        rgb_ext=args.rgb_ext,
        dsm_ext=args.dsm_ext,
    )