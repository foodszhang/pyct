#!/usr/bin/env python3
"""Inventory mouse CT and fluorescence acquisitions under true_data/ZJ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def numeric_tifs(path: Path) -> list[Path]:
    """Return projection TIFF files with numeric angle names."""
    result = []
    for item in path.glob("*.tif"):
        try:
            float(item.stem)
        except ValueError:
            continue
        result.append(item)
    return result


def flu_sessions(flu_dir: Path) -> list[dict]:
    """Summarize fluorescence exposure subdirectories."""
    sessions = []
    for path in sorted((p for p in flu_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        processed = list((path / "processed_tiff").glob("*.tif*"))
        npz = list((path / "npz").glob("*.npz"))
        spe = [p for p in path.glob("*.spe") if not p.stem.endswith("-raw")]
        sessions.append(
            {
                "name": path.name,
                "processed_tiff_count": len(processed),
                "npz_count": len(npz),
                "spe_count": len(spe),
            }
        )
    return sessions


def main() -> int:
    """Write a stable machine-readable and CSV batch manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/foods/pro/true_data/ZJ"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "dataset_inventory"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for subject in sorted(
        (p for p in args.root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        projections = numeric_tifs(subject / "ct")
        sessions = flu_sessions(subject / "flu")
        reconstructions = sorted(p.name for p in subject.glob("*.nii.gz"))
        flu_frames = sum(max(s["processed_tiff_count"], s["npz_count"], s["spe_count"]) for s in sessions)
        status = "ready_for_ct_qc"
        if not projections:
            status = "missing_ct"
        elif len(projections) < 270:
            status = "review_ct_completeness"
        if not sessions or flu_frames == 0:
            status += "+missing_flu"
        row = {
            "subject": subject.name,
            "ct_projection_count": len(projections),
            "flu_session_count": len(sessions),
            "flu_frame_count": flu_frames,
            "reconstruction_count": len(reconstructions),
            "status": status,
        }
        rows.append(row)
        details.append({**row, "flu_sessions": sessions, "reconstructions": reconstructions})
    with open(output_dir / "mouse_dataset_inventory.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["subject"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "mouse_dataset_inventory.json").write_text(
        json.dumps({"version": 1, "subjects": details}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_dir / "mouse_dataset_inventory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
