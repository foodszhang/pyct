#!/usr/bin/env python3
"""Fit three phantom source priors from user-marked projection tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def fit_track(
    observations: list[dict[str, Any]],
    center_uv: np.ndarray,
    pixel_size_mm: float,
    volume_center: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit x/z from detector-u sinusoid and y from detector-v median."""
    matrix = np.asarray(
        [
            [np.cos(np.deg2rad(float(item["angle_deg"]))), np.sin(np.deg2rad(float(item["angle_deg"])))]
            for item in observations
        ],
        dtype=np.float64,
    )
    detector_u_mm = np.asarray(
        [(float(item["u_px"]) - center_uv[0]) * pixel_size_mm for item in observations],
        dtype=np.float64,
    )
    xz_centered, _, _, _ = np.linalg.lstsq(matrix, detector_u_mm, rcond=None)
    residual = matrix @ xz_centered - detector_u_mm
    y = volume_center[1] + np.median(
        [(float(item["v_px"]) - center_uv[1]) * pixel_size_mm for item in observations]
    )
    center = np.asarray(
        [
            volume_center[0] + xz_centered[0],
            y,
            volume_center[2] + xz_centered[1],
        ],
        dtype=np.float64,
    )
    return center, float(np.sqrt(np.mean(residual**2)))


def nearest_body_point(center: np.ndarray, labels: np.ndarray, spacing: float) -> tuple[np.ndarray, float, int]:
    """Project a center to the nearest labelled body voxel if needed."""
    voxel = np.floor(center / spacing).astype(int)
    if np.all(voxel >= 0) and np.all(voxel < np.asarray(labels.shape)) and labels[tuple(voxel)] > 0:
        return center, 0.0, int(labels[tuple(voxel)])
    points = (np.argwhere(labels > 0).astype(np.float64) + 0.5) * spacing
    distances = np.linalg.norm(points - center[None, :], axis=1)
    index = int(np.argmin(distances))
    nearest = points[index]
    nearest_voxel = np.floor(nearest / spacing).astype(int)
    return nearest, float(distances[index]), int(labels[tuple(nearest_voxel)])


def main() -> int:
    """Write tumor_params JSON from manual annotation tracks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-points", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--radius-mm", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    points = json.loads(args.annotation_points.read_text(encoding="utf-8"))
    preprocess = json.loads((args.sample_dir / "projection_preprocess.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    labels = np.asarray(nib.load(str(args.shared_dir / "body_labels.nii.gz")).dataobj, dtype=np.uint8)
    center_uv = np.asarray(preprocess["model_detector"]["center_uv_px_in_real_camera"], dtype=np.float64)
    pixel_size_mm = float(preprocess["camera_pixel_size_mm"])
    volume_center = np.asarray(manifest["volume_center_world_mm"], dtype=np.float64)
    spacing = float(manifest["mcx_volume"]["voxel_size_mm"])

    colors = ["red", "green", "purple"]
    foci = []
    diagnostics = []
    for color in colors:
        observations = points["tracks"].get(color, [])
        if len(observations) < 3:
            raise ValueError(f"Not enough observations for {color}: {len(observations)}")
        raw_center, fit_rms_mm = fit_track(observations, center_uv, pixel_size_mm, volume_center)
        center, clamp_distance_mm, tissue_label = nearest_body_point(raw_center, labels, spacing)
        foci.append(
            {
                "center": [round(float(value), 4) for value in center],
                "center_atlas_mm": None,
                "shape": "sphere",
                "params": {
                    "radius": args.radius_mm,
                    "source_type": "gaussian",
                    "intensity": 1.0,
                    "placement_tissue_label": tissue_label,
                },
                "radius": args.radius_mm,
                "rx": None,
                "ry": None,
                "rz": None,
                "name": f"user_marked_{color}_source",
            }
        )
        diagnostics.append(
            {
                "color": color,
                "observation_count": len(observations),
                "raw_center_world_mm": [round(float(value), 4) for value in raw_center],
                "center_world_mm": [round(float(value), 4) for value in center],
                "fit_rms_u_mm": round(fit_rms_mm, 4),
                "body_clamp_distance_mm": round(clamp_distance_mm, 4),
                "placement_tissue_label": tissue_label,
            }
        )

    params = {
        "num_foci": len(foci),
        "depth_tier": "real_phantom_user_manual_annotation",
        "depth_mm": None,
        "source_type": "gaussian",
        "foci": foci,
        "organ_constraint_passed": True,
        "gt_status": "user_marked_projection_fit_estimated_prior",
        "fit_variant": {
            "type": "phantom_manual_projection_annotation",
            "annotation_points": str(args.annotation_points.resolve()),
            "fit_diagnostics": diagnostics,
        },
    }
    output = args.output or args.sample_dir / "tumor_params_user_marked.json"
    output.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
