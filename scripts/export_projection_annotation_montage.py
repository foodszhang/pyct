#!/usr/bin/env python3
"""Export full-camera fluorescence frames as a PNG suitable for manual marks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize one frame independently while retaining low-level reflections."""
    lo, hi = np.percentile(image, [0.5, 99.8])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """Write a labelled five-column montage from a full-resolution projection archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Mark real source centers; use a different color for reflection")
    parser.add_argument("--tile-width", type=int, default=384)
    args = parser.parse_args()

    with np.load(args.input) as archive:
        angles = sorted((int(key) for key in archive.files), reverse=True)
        images = [archive[str(angle)].astype(np.float32) for angle in angles]
    tiles = []
    for angle, image in zip(angles, images):
        scale = args.tile_width / image.shape[1]
        tile = cv2.resize(image, (args.tile_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(normalize(tile), cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, f"{angle:+d} deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
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
    if not cv2.imwrite(str(args.output), np.vstack([header, montage])):
        raise RuntimeError(f"Failed to write {args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
