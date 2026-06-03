#!/usr/bin/env python3
"""Extract manually marked fluorescence bead centers from binary TIFF masks."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected foreground components as lists of (y, x) pixels."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            py, px = queue.popleft()
            pixels.append((py, px))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        components.append(pixels)
    return components


def component_row(mask_name: str, component_id: int, pixels: list[tuple[int, int]]) -> dict[str, object]:
    """Summarize one connected component."""
    arr = np.asarray(pixels, dtype=np.int32)
    ys = arr[:, 0]
    xs = arr[:, 1]
    return {
        "mask": mask_name,
        "component": component_id,
        "area_px": len(pixels),
        "center_x_px": round(float(xs.mean()), 3),
        "center_y_px": round(float(ys.mean()), 3),
        "min_x_px": int(xs.min()),
        "min_y_px": int(ys.min()),
        "max_x_px": int(xs.max()),
        "max_y_px": int(ys.max()),
    }


def find_background(mask_path: Path) -> Image.Image | None:
    """Find the processed enhanced PNG matching a short mask name."""
    sample_id = mask_path.stem.removesuffix("_Mask")
    matches = sorted(mask_path.parent.glob(f"* {sample_id}_processed_enhanced.png"))
    if not matches:
        return None
    return Image.open(matches[0]).convert("RGB")


def annotate(mask_path: Path, rows: list[dict[str, object]], output_path: Path) -> None:
    """Write an annotated preview, using the enhanced source image when available."""
    mask = Image.open(mask_path).convert("L")
    preview = find_background(mask_path)
    if preview is None:
        preview = Image.new("RGB", mask.size, color=(0, 0, 0))
        preview.paste((180, 180, 180), mask=mask)
    draw = ImageDraw.Draw(preview)
    for row in rows:
        x = float(row["center_x_px"])
        y = float(row["center_y_px"])
        label = str(row["component"])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0), width=2)
        draw.text((x + 10, y - 8), label, fill=(255, 255, 0))
    preview.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-area", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or args.mask_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for mask_path in sorted(args.mask_dir.glob("*_Mask.tif")):
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        components = connected_components(mask)
        rows: list[dict[str, object]] = []
        for pixels in components:
            if len(pixels) < args.min_area:
                continue
            rows.append(component_row(mask_path.name, len(rows) + 1, pixels))
        rows.sort(key=lambda row: float(row["center_y_px"]))
        for component_id, row in enumerate(rows, start=1):
            row["component"] = component_id
        all_rows.extend(rows)
        annotate(mask_path, rows, output_dir / f"{mask_path.stem}_points.png")
        print(f"{mask_path.name}: {len(rows)} point(s)")

    csv_path = output_dir / "fluorescence_mask_points.csv"
    fieldnames = [
        "mask",
        "component",
        "area_px",
        "center_x_px",
        "center_y_px",
        "min_x_px",
        "min_y_px",
        "max_x_px",
        "max_y_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
