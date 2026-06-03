#!/usr/bin/env python3
"""Generate body-constrained mouse source candidates from a rotating real hotspot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from import_real_zj_to_fmt_simgen import MODEL_ANGLES, build_model_projections


ANGLES = list(range(-90, 91, 10))


def strongest_centroid(image: np.ndarray) -> tuple[float, float]:
    """Return centroid of the strongest smoothed top-percentile component."""
    smooth = ndimage.gaussian_filter(np.clip(image.astype(np.float32), 0.0, None), sigma=3.0)
    labels, count = ndimage.label(smooth >= np.percentile(smooth, 99.0))
    if count == 0:
        raise RuntimeError("No hotspot component found")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    vv, uu = np.nonzero(labels == int(np.argmax(sizes)))
    weights = smooth[vv, uu]
    return float(np.sum(uu * weights) / np.sum(weights)), float(np.sum(vv * weights) / np.sum(weights))


def safe_scale_name(scale: float) -> str:
    """Encode a scale for filenames."""
    return f"{scale:.3f}".replace(".", "p")


def main() -> int:
    """Fit detector trajectory and write scale/angle-offset source candidates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--center-u-px", type=float, required=True)
    parser.add_argument("--center-v-px", type=float, required=True)
    parser.add_argument("--pixel-scales-mm", type=float, nargs="+", default=[0.12, 0.16, 0.20])
    parser.add_argument("--allowed-labels", type=int, nargs="+", default=[1, 6, 7, 8, 9])
    parser.add_argument("--radius-mm", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = args.sample_dir / "real_source_fit_candidates"
    output_dir.mkdir(exist_ok=True)
    with np.load(args.sample_dir / "proj_real_corrected_full.npz") as archive:
        corrected = {key: archive[key].astype(np.float32) for key in archive.files}
    centroids = np.asarray([strongest_centroid(corrected[str(angle)]) for angle in ANGLES])
    detector_u_px = centroids[:, 0] - args.center_u_px
    matrix = np.asarray(
        [[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))] for angle in ANGLES],
        dtype=np.float64,
    )
    xz_px, _, _, _ = np.linalg.lstsq(matrix, detector_u_px, rcond=None)
    labels = np.asarray(nib.load(str(args.shared_dir / "body_labels.nii.gz")).dataobj, dtype=np.uint8)
    manifest = json.loads((args.shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    spacing = float(manifest["mcx_volume"]["voxel_size_mm"])
    volume_center = np.asarray(manifest["volume_center_world_mm"], dtype=np.float64)
    tissue_points = (np.argwhere(np.isin(labels, args.allowed_labels)).astype(np.float64) + 0.5) * spacing
    results = []
    for scale in args.pixel_scales_mm:
        target_y = volume_center[1] + float(np.median(centroids[:, 1] - args.center_v_px)) * scale
        best = None
        for offset_deg in range(-180, 180, 5):
            angle = np.deg2rad(offset_deg)
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            target_xz = volume_center[[0, 2]] + rotation @ (xz_px * scale)
            target = np.asarray([target_xz[0], target_y, target_xz[1]])
            distances = np.linalg.norm(tissue_points - target[None, :], axis=1)
            index = int(np.argmin(distances))
            item = (float(distances[index]), offset_deg, target, tissue_points[index])
            if best is None or item[0] < best[0]:
                best = item
        distance_mm, offset_deg, target, center = best
        voxel = np.floor(center / spacing).astype(int)
        name = safe_scale_name(scale)
        candidate_dir = output_dir / f"scale_{name}"
        candidate_dir.mkdir(exist_ok=True)
        params = {
            "num_foci": 1,
            "depth_tier": "real_mouse_initial_prior",
            "depth_mm": None,
            "source_type": "gaussian",
            "foci": [
                {
                    "center": [round(float(value), 4) for value in center],
                    "center_atlas_mm": None,
                    "shape": "sphere",
                    "params": {
                        "radius": args.radius_mm,
                        "source_type": "gaussian",
                        "intensity": 1.0,
                        "placement_tissue_label": int(labels[tuple(voxel)]),
                    },
                    "radius": args.radius_mm,
                    "rx": None,
                    "ry": None,
                    "rz": None,
                    "name": "estimated_real_mouse_focus",
                }
            ],
            "organ_constraint_passed": True,
            "gt_status": "initial_prior_requires_mcx_projection_fit",
            "fit_variant": {
                "type": "mouse_real_scale_angle_offset_candidate",
                "camera_pixel_size_mm": scale,
                "angle_offset_deg": offset_deg,
                "unconstrained_target_world_mm": target.tolist(),
                "body_constraint_distance_mm": distance_mm,
            },
        }
        (candidate_dir / "tumor_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        model = build_model_projections(
            corrected,
            angles=MODEL_ANGLES,
            center_u_px=args.center_u_px,
            center_v_px=args.center_v_px,
            camera_pixel_size_mm=scale,
            fov_mm=80.0,
            output_resolution=256,
        )
        np.savez_compressed(candidate_dir / "proj_real_scale_candidate.npz", **model)
        results.append({"scale": scale, "angle_offset_deg": offset_deg, "center": center.tolist(), "distance_mm": distance_mm})
    (output_dir / "candidate_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
