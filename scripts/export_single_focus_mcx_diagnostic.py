#!/usr/bin/env python3
"""Export a real-versus-multi-versus-single-focus MCX diagnostic montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_real_mcx_projections import ANGLES, normalized


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--single-focus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def projection_path_for_focus(root: Path, focus_index: int) -> Path:
    """Find the isolated MCX projection for one diagnostic focus."""
    for candidate_dir in root.glob("candidate_*"):
        params_path = candidate_dir / "tumor_params.json"
        projection_path = candidate_dir / "proj_mcx_candidate.npz"
        if not params_path.exists() or not projection_path.exists():
            continue
        params = json.loads(params_path.read_text(encoding="utf-8"))
        variant = params.get("fit_variant", {})
        if variant.get("type") == "single_focus_diagnostic" and variant.get("focus_index") == focus_index:
            return projection_path
    raise FileNotFoundError(f"No single-focus MCX projection found for focus {focus_index}")


def main() -> int:
    """Write montage and per-focus visible-output statistics."""
    args = parse_args()
    single_paths = [projection_path_for_focus(args.single_focus_root, index) for index in range(1, 4)]
    report = {"version": 1, "angles": ANGLES, "single_focus": {}}
    rows = []
    with np.load(args.real) as real, np.load(args.combined) as combined:
        singles = [np.load(path) for path in single_paths]
        try:
            for angle in ANGLES:
                tiles = []
                for name, image in [
                    ("real", real[str(angle)]),
                    ("combined", combined[str(angle)]),
                    ("focus_1", singles[0][str(angle)]),
                    ("focus_2", singles[1][str(angle)]),
                    ("focus_3", singles[2][str(angle)]),
                ]:
                    tile = (normalized(image, 2.0) * 255.0).astype(np.uint8)
                    tile = cv2.resize(tile, (192, 192), interpolation=cv2.INTER_AREA)
                    cv2.putText(tile, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 255, 1)
                    tiles.append(tile)
                row = np.hstack(tiles)
                cv2.putText(row, f"{angle:+d}", (6, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.52, 255, 1)
                rows.append(row)
            for index, archive in enumerate(singles, 1):
                report["single_focus"][str(index)] = {
                    str(angle): {
                        "max": float(archive[str(angle)].max()),
                        "sum": float(archive[str(angle)].sum()),
                    }
                    for angle in ANGLES
                }
        finally:
            for archive in singles:
                archive.close()
    cv2.imwrite(str(args.output), np.vstack(rows))
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
