#!/usr/bin/env python3
"""Export a LightField SPE acquisition as a manual annotation montage."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import spe_loader as sl
except ModuleNotFoundError:
    for candidate in (
        Path("/home/foods/pro/true_data/ZJ/.venv/lib/python3.12/site-packages"),
        Path("/home/foods/pro/mcx_simulation/.venv/lib/python3.12/site-packages"),
    ):
        if (candidate / "spe_loader.py").exists():
            sys.path.insert(0, str(candidate))
            import spe_loader as sl

            break
    else:
        raise


INDEX_RE = re.compile(r" (\d+)(-raw)?\.spe$", re.IGNORECASE)


def load_stack(path: Path) -> np.ndarray:
    """Load the first SPE ROI as (frames, height, width)."""
    spe = sl.SpeFile(str(path))
    if not spe.data or not spe.data[0]:
        raise ValueError(f"Empty SPE file: {path}")
    return np.stack([np.asarray(frame[0]) for frame in spe.data], axis=0).astype(np.float32)


def group_processed(input_dir: Path) -> dict[int, Path]:
    """Return processed SPE files keyed by acquisition index."""
    grouped: dict[int, Path] = {}
    for path in input_dir.glob("*.spe"):
        match = INDEX_RE.search(path.name)
        if match is None or match.group(2):
            continue
        grouped[int(match.group(1))] = path
    return grouped


def normalize(image: np.ndarray) -> np.ndarray:
    """Robustly normalize one fluorescence frame."""
    lo, hi = np.percentile(image, [0.5, 99.8])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """Write a 5-column +90..-90 annotation montage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-angle", type=int, default=-90)
    parser.add_argument("--angle-step", type=int, default=10)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--title", default="Phantom: mark source 1/2/3 centers separately")
    args = parser.parse_args()

    grouped = group_processed(args.input_dir)
    records = []
    for position, index in enumerate(sorted(grouped)):
        angle = args.first_angle + position * args.angle_step
        frame = np.median(load_stack(grouped[index]), axis=0)
        records.append((angle, index, frame))
    records.sort(key=lambda item: item[0], reverse=True)
    if not records:
        raise ValueError(f"No processed SPE files found in {args.input_dir}")

    tiles = []
    for angle, index, frame in records:
        scale = args.tile_width / frame.shape[1]
        tile = cv2.resize(frame, (args.tile_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(normalize(tile), cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, f"{angle:+d} deg  idx={index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    columns = 5
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    montage = np.vstack(rows)
    header = np.zeros((54, montage.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, args.title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack([header, montage]))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
