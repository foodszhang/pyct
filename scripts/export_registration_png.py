#!/usr/bin/env python3
"""Export fluorescence and CT calibration images for manual registration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import spe_loader as sl
from PIL import Image, ImageDraw


def percentile_uint8(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Convert an image to uint8 using percentile contrast stretching."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def add_grid(img: Image.Image, label: str, step: int = 50) -> Image.Image:
    """Overlay a pixel-coordinate grid on an image."""
    rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size

    for x in range(0, width, step):
        draw.line((x, 0, x, height - 1), fill=(255, 255, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, height, step):
        draw.line((0, y, width - 1, y), fill=(255, 255, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 0))

    draw.rectangle((0, height - 22, width - 1, height - 1), fill=(0, 0, 0))
    draw.text((5, height - 19), label, fill=(255, 255, 255))
    return rgb


def load_first_spe_frame(path: Path) -> np.ndarray:
    """Load the first ROI from the first frame of a LightField SPE file."""
    spe = sl.SpeFile(str(path))
    return np.asarray(spe.data[0][0])


def save_variants(img: np.ndarray, output_dir: Path, stem: str) -> Image.Image:
    """Save enhanced grayscale and coordinate-grid variants."""
    enhanced = Image.fromarray(percentile_uint8(img))
    enhanced.save(output_dir / f"{stem}_enhanced.png")
    grid = add_grid(enhanced, stem)
    grid.save(output_dir / f"{stem}_grid.png")
    return grid


def export_fluorescence(input_dir: Path, output_dir: Path) -> None:
    """Export processed and raw SPE calibration images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.spe")):
        suffix = "_raw" if path.stem.endswith("-raw") else "_processed"
        stem = path.stem.removesuffix("-raw")
        image = load_first_spe_frame(path)
        save_variants(image, output_dir, f"{stem}{suffix}")


def export_ct(input_dir: Path, output_dir: Path, angles: list[int]) -> None:
    """Export selected CT projections and a contact sheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grids: list[Image.Image] = []
    for angle in angles:
        path = input_dir / f"{angle}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        image = np.asarray(Image.open(path))
        grids.append(save_variants(image, output_dir, f"ct_{angle:03d}deg"))

    thumb_width = 384
    thumbs: list[Image.Image] = []
    for grid in grids:
        thumb_height = round(grid.height * thumb_width / grid.width)
        thumbs.append(grid.resize((thumb_width, thumb_height)))

    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_width * columns, thumbs[0].height * rows))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb.height))
    sheet.save(output_dir / "ct_key_angles_overview.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flu-dir", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ct-angles",
        type=int,
        nargs="+",
        default=[0, 45, 90, 135, 180, 225, 270, 315],
    )
    args = parser.parse_args()

    export_fluorescence(args.flu_dir, args.output_dir / "fluorescence")
    export_ct(args.ct_dir, args.output_dir / "ct", args.ct_angles)
    print(f"Exported registration PNGs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
