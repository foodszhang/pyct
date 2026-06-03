#!/usr/bin/env python3
"""Extract user-marked phantom source tracks from the annotation montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


COLOR_RANGES = {
    "red": [([0, 90, 90], [10, 255, 255]), ([170, 90, 90], [179, 255, 255])],
    "green": [([40, 70, 70], [85, 255, 255])],
    "purple": [([125, 45, 45], [165, 255, 255])],
}


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Return a binary mask for one annotation color."""
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8)) > 0
    return ndimage.binary_closing(mask, iterations=1)


def extract_components(mask: np.ndarray, min_pixels: int) -> list[tuple[float, float, int]]:
    """Extract connected colored strokes as image-space centroids."""
    labels, count = ndimage.label(mask)
    items = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) < min_pixels:
            continue
        items.append((float(xx.mean()), float(yy.mean()), int(len(xx))))
    return items


def main() -> int:
    """Map colored montage circles back to full-camera projection coordinates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--projection-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-height", type=int, default=54)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--min-pixels", type=int, default=40)
    args = parser.parse_args()

    image = cv2.imread(str(args.annotation), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.annotation)
    with np.load(args.projection_full) as archive:
        angles = sorted((int(key) for key in archive.files), reverse=True)
        frame_shape = archive[str(angles[0])].shape

    tile_width = image.shape[1] // args.columns
    tile_height = (image.shape[0] - args.header_height) // int(np.ceil(len(angles) / args.columns))
    scale_x = frame_shape[1] / tile_width
    scale_y = frame_shape[0] / tile_height
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    tracks: dict[str, list[dict[str, float | int | str]]] = {}
    for color in COLOR_RANGES:
        observations = []
        for x_img, y_img, pixels in extract_components(color_mask(hsv, color), args.min_pixels):
            if y_img < args.header_height:
                continue
            col = int(x_img // tile_width)
            row = int((y_img - args.header_height) // tile_height)
            index = row * args.columns + col
            if index < 0 or index >= len(angles):
                continue
            local_x = x_img - col * tile_width
            local_y = y_img - args.header_height - row * tile_height
            observations.append(
                {
                    "angle_deg": int(angles[index]),
                    "u_px": round(float(local_x * scale_x), 3),
                    "v_px": round(float(local_y * scale_y), 3),
                    "annotation_pixels": pixels,
                }
            )
        observations.sort(key=lambda item: -int(item["angle_deg"]))
        tracks[color] = observations

    result = {
        "version": 1,
        "source": str(args.annotation.resolve()),
        "projection_full": str(args.projection_full.resolve()),
        "frame_shape_hw": list(frame_shape),
        "montage": {
            "tile_width_px": tile_width,
            "tile_height_px": tile_height,
            "header_height_px": args.header_height,
            "columns": args.columns,
        },
        "tracks": tracks,
        "color_semantics": {
            "red": "user_marked_phantom_source_track",
            "green": "user_marked_phantom_source_track",
            "purple": "user_marked_phantom_source_track",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
