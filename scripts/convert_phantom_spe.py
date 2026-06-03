#!/usr/bin/env python3
"""Convert timestamped phantom LightField SPE pairs into numbered NPZ and PNG files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import spe_loader as sl
from PIL import Image, ImageDraw


INDEX_RE = re.compile(r" (\d+)(-raw)?\.spe$", re.IGNORECASE)


def load_stack(path: Path) -> np.ndarray:
    """Load the first SPE ROI as a (frames, height, width) stack."""
    spe = sl.SpeFile(str(path))
    if not spe.data or not spe.data[0]:
        raise ValueError(f"Empty SPE file: {path}")
    return np.stack([np.asarray(frame[0]) for frame in spe.data], axis=0)


def percentile_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 with robust contrast stretching."""
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.8])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def save_contact_sheet(images: list[tuple[int, Image.Image]], output: Path) -> None:
    """Save numbered thumbnails in acquisition order."""
    columns = 5
    thumb_width = 320
    thumbs: list[Image.Image] = []
    for index, image in images:
        height = round(image.height * thumb_width / image.width)
        thumb = image.resize((thumb_width, height)).convert("RGB")
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((0, 0, 150, 24), fill=(0, 0, 0))
        draw.text((5, 5), f"index={index}", fill=(255, 255, 255))
        thumbs.append(thumb)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumbs[0].height))
    for pos, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((pos % columns) * thumb_width, (pos // columns) * thumb.height))
    sheet.save(output)


def group_spe_files(input_dir: Path) -> dict[int, dict[str, Path]]:
    """Pair timestamped processed and raw files by their trailing numeric index."""
    grouped: dict[int, dict[str, Path]] = {}
    for path in input_dir.glob("*.spe"):
        match = INDEX_RE.search(path.name)
        if match is None:
            raise ValueError(f"Cannot parse trailing index from {path.name}")
        index = int(match.group(1))
        kind = "raw" if match.group(2) else "processed"
        grouped.setdefault(index, {})[kind] = path
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--first-angle", type=int, default=-90)
    parser.add_argument("--angle-step", type=int, default=10)
    args = parser.parse_args()

    grouped = group_spe_files(args.input_dir)
    output_npz = args.input_dir / "npz"
    output_png = args.input_dir / "png"
    output_npz.mkdir(exist_ok=True)
    output_png.mkdir(exist_ok=True)

    manifest = []
    preview_images: list[tuple[int, Image.Image]] = []
    for position, index in enumerate(sorted(grouped)):
        pair = grouped[index]
        if "processed" not in pair:
            raise ValueError(f"Missing processed SPE for index {index}")
        processed = load_stack(pair["processed"])
        raw = load_stack(pair["raw"]) if "raw" in pair else None
        angle = args.first_angle + position * args.angle_step
        arrays = {
            "processed": processed,
            "angle_deg": np.asarray(angle, dtype=np.int32),
            "source_processed": pair["processed"].name,
        }
        if raw is not None:
            arrays["raw"] = raw
            arrays["source_raw"] = pair["raw"].name
        np.savez_compressed(output_npz / f"{index}.npz", **arrays)

        fused = np.median(processed.astype(np.float32), axis=0)
        preview = Image.fromarray(percentile_uint8(fused))
        preview.save(output_png / f"{index:02d}_{angle:+04d}deg_processed_enhanced.png")
        preview_images.append((index, preview))
        manifest.append(
            {
                "index": index,
                "angle_deg": angle,
                "processed_frames": int(processed.shape[0]),
                "processed_shape": list(processed.shape),
                "raw_frames": int(raw.shape[0]) if raw is not None else 0,
                "raw_shape": list(raw.shape) if raw is not None else None,
                "fusion": "pixelwise_median" if processed.shape[0] > 1 else "single_frame",
            }
        )

    (args.input_dir / "projection_manifest.json").write_text(
        json.dumps({"projections": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_contact_sheet(preview_images, output_png / "overview.png")
    print(f"[Done] converted {len(manifest)} projection pairs in {args.input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
