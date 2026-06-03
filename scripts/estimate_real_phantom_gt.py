#!/usr/bin/env python3
"""Estimate adjustable 3D fluorescence-source priors from real turntable images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage


TRACKS = [
    {
        "name": "source_1",
        "window_uvuv": [300, 530, 220, 270],
        "min_peak_ratio": 0.18,
        "confidence": "high",
        "fit_angles": None,
    },
    {
        "name": "source_2",
        "window_uvuv": [280, 450, 335, 390],
        "min_peak_ratio": 0.18,
        "confidence": "medium",
        "fit_angles": None,
    },
    {
        "name": "source_3",
        "window_uvuv": [440, 520, 270, 315],
        "min_peak_ratio": 0.18,
        "confidence": "low_reflection_overlap",
        "fit_angles": list(range(20, 91, 10)),
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--source-radius-mm", type=float, default=2.0)
    return parser.parse_args()


def load_manifest_center(shared_dir: Path) -> np.ndarray:
    """Load subject rotation center from the authoritative frame manifest."""
    manifest = json.loads((shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    return np.asarray(manifest["volume_center_world_mm"], dtype=np.float64)


def component_centroid(image: np.ndarray, window: list[int]) -> dict[str, float]:
    """Locate the strongest smoothed component inside one source-specific window."""
    u0, u1, v0, v1 = window
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=2.0)
    crop = smoothed[v0:v1, u0:u1]
    threshold = float(np.percentile(crop, 92.0))
    labels, count = ndimage.label(crop >= threshold)
    if count == 0:
        raise ValueError(f"No fluorescence component found in window {window}")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label = int(np.argmax(sizes))
    vv, uu = np.nonzero(labels == label)
    weights = crop[vv, uu].astype(np.float64)
    return {
        "u_px": float(np.sum((uu + u0) * weights) / np.sum(weights)),
        "v_px": float(np.sum((vv + v0) * weights) / np.sum(weights)),
        "peak": float(weights.max()),
        "component_pixels": int(len(uu)),
    }


def estimate_track(
    angles: list[int],
    images: np.ndarray,
    track: dict[str, Any],
    camera_center_uv: np.ndarray,
    camera_pixel_size_mm: float,
    volume_center_world: np.ndarray,
) -> dict[str, Any]:
    """Fit x*cos(theta)+z*sin(theta) and vertical position from one image track."""
    observations = []
    for angle, image in zip(angles, images):
        observation = component_centroid(image, track["window_uvuv"])
        observation["angle_deg"] = int(angle)
        observations.append(observation)

    max_peak = max(float(item["peak"]) for item in observations)
    fit_angles = track["fit_angles"]
    selected = [
        item
        for item in observations
        if item["peak"] >= max_peak * float(track["min_peak_ratio"])
        and (fit_angles is None or item["angle_deg"] in fit_angles)
    ]
    matrix = np.asarray(
        [
            [np.cos(np.deg2rad(item["angle_deg"])), np.sin(np.deg2rad(item["angle_deg"]))]
            for item in selected
        ],
        dtype=np.float64,
    )
    detector_u_mm = np.asarray(
        [(item["u_px"] - camera_center_uv[0]) * camera_pixel_size_mm for item in selected],
        dtype=np.float64,
    )
    xz_centered, _, _, _ = np.linalg.lstsq(matrix, detector_u_mm, rcond=None)
    predicted = matrix @ xz_centered
    rms_mm = float(np.sqrt(np.mean((predicted - detector_u_mm) ** 2)))
    y_world = float(
        volume_center_world[1]
        + np.median(
            [(item["v_px"] - camera_center_uv[1]) * camera_pixel_size_mm for item in selected]
        )
    )
    center_world = [
        float(volume_center_world[0] + xz_centered[0]),
        y_world,
        float(volume_center_world[2] + xz_centered[1]),
    ]
    selected_angles = {int(item["angle_deg"]) for item in selected}
    for item in observations:
        item["used_for_fit"] = int(item["angle_deg"]) in selected_angles
    return {
        "name": track["name"],
        "confidence": track["confidence"],
        "window_uvuv": track["window_uvuv"],
        "center_world_mm": [round(value, 4) for value in center_world],
        "fit_rms_mm": round(rms_mm, 4),
        "selected_angles_deg": sorted(selected_angles),
        "observations": observations,
    }


def save_diagnostic(
    path: Path,
    angles: list[int],
    images: np.ndarray,
    estimates: list[dict[str, Any]],
) -> None:
    """Save a montage with detected source centers and search windows."""
    colors = [(255, 255, 255), (255, 180, 80), (80, 180, 255)]
    tiles = []
    for index, (angle, image) in enumerate(zip(angles, images)):
        lo, hi = np.percentile(image, [1.0, 99.7])
        tile = np.clip((image - lo) / (hi - lo + 1.0e-8) * 255.0, 0, 255).astype(np.uint8)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        for color, estimate in zip(colors, estimates):
            observation = estimate["observations"][index]
            u0, u1, v0, v1 = estimate["window_uvuv"]
            cv2.rectangle(tile, (u0, v0), (u1, v1), color, 1)
            center = (int(round(observation["u_px"])), int(round(observation["v_px"])))
            cv2.drawMarker(tile, center, color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(tile, f"{angle:+d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        tiles.append(cv2.resize(tile, (320, 256), interpolation=cv2.INTER_AREA))
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), 5):
        row = tiles[start : start + 5]
        rows.append(np.hstack(row + [blank] * (5 - len(row))))
    cv2.imwrite(str(path), np.vstack(rows))


def evaluate_gaussian_sources(points_mm: np.ndarray, foci: list[dict[str, Any]]) -> np.ndarray:
    """Evaluate max-fused Gaussian source priors at XYZ world coordinates."""
    values = np.zeros(len(points_mm), dtype=np.float32)
    for focus in foci:
        center = np.asarray(focus["center"], dtype=np.float32)
        radius = float(focus["radius"])
        intensity = float(focus["params"]["intensity"])
        distance = np.linalg.norm(points_mm - center[None, :], axis=1)
        source = np.exp(-0.5 * (distance / radius) ** 2).astype(np.float32) * intensity
        source[distance > 3.0 * radius] = 0.0
        values = np.maximum(values, source)
    return values


def save_estimated_gt(sample_dir: Path, shared_dir: Path, foci: list[dict[str, Any]]) -> None:
    """Save estimated voxel and mesh-node GT arrays without claiming measured truth."""
    manifest = json.loads((shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    shape_xyz = tuple(int(value) for value in manifest["mcx_volume"]["shape_xyz"])
    voxel_size_mm = float(manifest["mcx_volume"]["voxel_size_mm"])
    grid = np.stack(
        np.meshgrid(
            np.arange(shape_xyz[0], dtype=np.float32),
            np.arange(shape_xyz[1], dtype=np.float32),
            np.arange(shape_xyz[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    points_mm = (grid.reshape(-1, 3) + 0.5) * voxel_size_mm
    gt_voxels = evaluate_gaussian_sources(points_mm, foci).reshape(shape_xyz)
    np.save(sample_dir / "gt_voxels_estimated.npy", gt_voxels.astype(np.float32))

    mesh = np.load(shared_dir / "mesh.npz")
    gt_nodes = evaluate_gaussian_sources(mesh["nodes"].astype(np.float32), foci)
    np.save(sample_dir / "gt_nodes_estimated.npy", gt_nodes.astype(np.float32))


def main() -> int:
    """Estimate initial GT and write an editable MCX tumor_params file."""
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    preprocess = json.loads((sample_dir / "projection_preprocess.json").read_text(encoding="utf-8"))
    camera_center_uv = np.asarray(
        preprocess["model_detector"]["center_uv_px_in_real_camera"],
        dtype=np.float64,
    )
    camera_pixel_size_mm = float(preprocess["camera_pixel_size_mm"])
    volume_center_world = load_manifest_center(args.shared_dir.resolve())

    with np.load(sample_dir / "proj_real_corrected_full.npz") as archive:
        angles = [int(key) for key in archive.files]
        images = np.stack([archive[str(angle)].astype(np.float32) for angle in angles], axis=0)

    estimates = [
        estimate_track(
            angles,
            images,
            track,
            camera_center_uv,
            camera_pixel_size_mm,
            volume_center_world,
        )
        for track in TRACKS
    ]
    estimate_path = sample_dir / "real_gt_initial_estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "initial_prior_requires_mcx_projection_fit",
                "coordinate_frame": "mcx_trunk_local_mm",
                "camera_center_uv_px": camera_center_uv.tolist(),
                "camera_pixel_size_mm": camera_pixel_size_mm,
                "tracks": estimates,
                "notes": [
                    "source_1 and source_2 are fitted from stable rotating tracks.",
                    "source_3 overlaps angle-dependent right-side reflection and is a low-confidence initial value.",
                    "Refine center, radius, and intensity by MCX projection similarity before using as formal GT.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    foci = []
    for index, estimate in enumerate(estimates, 1):
        foci.append(
            {
                "center": estimate["center_world_mm"],
                "center_atlas_mm": None,
                "shape": "sphere",
                "params": {
                    "radius": args.source_radius_mm,
                    "source_type": "gaussian",
                    "intensity": 1.0,
                    "fit_confidence": estimate["confidence"],
                },
                "radius": args.source_radius_mm,
                "rx": None,
                "ry": None,
                "rz": None,
                "name": f"estimated_source_{index}",
            }
        )
    tumor_params = {
        "num_foci": len(foci),
        "depth_tier": "real_phantom_initial_prior",
        "depth_mm": None,
        "source_type": "gaussian",
        "foci": foci,
        "organ_constraint_passed": True,
        "gt_status": "initial_prior_requires_mcx_projection_fit",
        "estimate_source": str(estimate_path),
    }
    (sample_dir / "tumor_params.json").write_text(
        json.dumps(tumor_params, indent=2),
        encoding="utf-8",
    )
    save_estimated_gt(sample_dir, args.shared_dir.resolve(), foci)
    save_diagnostic(sample_dir / "real_gt_track_diagnostic.png", angles, images, estimates)

    print(f"Wrote initial GT estimate: {estimate_path}")
    print(f"Wrote editable MCX source prior: {sample_dir / 'tumor_params.json'}")
    for estimate in estimates:
        print(
            f"{estimate['name']}: center={estimate['center_world_mm']} "
            f"rms={estimate['fit_rms_mm']}mm confidence={estimate['confidence']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
