#!/usr/bin/env python3
"""Generate artificial mouse sources for MCX train/validation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def evaluate_gaussian_sources(points_mm: np.ndarray, foci: list[dict]) -> np.ndarray:
    """Evaluate max-fused Gaussian source priors at XYZ coordinates."""
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


def sample_centers(
    labels: np.ndarray,
    spacing_mm: float,
    rng: np.random.Generator,
    n_foci: int,
    allowed_labels: list[int],
) -> list[dict]:
    """Sample separated source centers inside tissue with a body-boundary margin."""
    body_distance_mm = ndimage.distance_transform_edt(labels > 0) * spacing_mm
    foci = []
    for index in range(n_foci):
        radius = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
        intensity = float(rng.uniform(0.55, 1.45))
        candidates = np.argwhere(
            np.isin(labels, allowed_labels) & (body_distance_mm >= min(3.0 * radius, 3.0))
        )
        rng.shuffle(candidates)
        center = None
        tissue_label = None
        for voxel in candidates:
            candidate = (voxel.astype(np.float32) + 0.5) * spacing_mm
            if all(np.linalg.norm(candidate - np.asarray(focus["center"])) >= radius + float(focus["radius"]) + 2.0 for focus in foci):
                center = candidate
                tissue_label = int(labels[tuple(voxel)])
                break
        if center is None:
            raise RuntimeError(f"Could not place focus {index + 1}/{n_foci}")
        foci.append(
            {
                "center": [round(float(value), 4) for value in center],
                "center_atlas_mm": None,
                "shape": "sphere",
                "params": {
                    "radius": radius,
                    "source_type": "gaussian",
                    "intensity": round(intensity, 6),
                    "placement_tissue_label": tissue_label,
                },
                "radius": radius,
                "rx": None,
                "ry": None,
                "rz": None,
                "name": f"artificial_focus_{index + 1}",
            }
        )
    return foci


def write_split(path: Path, samples: list[str]) -> None:
    """Write one sample ID per line."""
    path.write_text("".join(f"{sample}\n" for sample in samples), encoding="utf-8")


def main() -> int:
    """Generate GT arrays and split files for a small or full simulation batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--allowed-labels", type=int, nargs="+", default=[1, 6, 7, 8, 9])
    args = parser.parse_args()
    shared_dir = args.bundle_root / "subjects" / args.subject_id / "shared"
    samples_dir = args.bundle_root / "samples"
    split_dir = args.bundle_root / "splits"
    samples_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    labels = np.asarray(nib.load(str(shared_dir / "body_labels.nii.gz")).dataobj, dtype=np.uint8)
    spacing = float(manifest["mcx_volume"]["voxel_size_mm"])
    mesh = np.load(shared_dir / "mesh.npz")
    mesh_nodes = mesh["nodes"].astype(np.float32)
    grid = np.stack(
        np.meshgrid(
            np.arange(labels.shape[0], dtype=np.float32),
            np.arange(labels.shape[1], dtype=np.float32),
            np.arange(labels.shape[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    voxel_points = (grid.reshape(-1, 3) + 0.5) * spacing
    rng = np.random.default_rng(args.seed)
    sample_names = []
    for index in range(args.count):
        sample_name = f"sample_{args.subject_id}_sim_{index:04d}"
        sample_dir = samples_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        n_foci = 1 + index % 3
        foci = sample_centers(labels, spacing, rng, n_foci, args.allowed_labels)
        params = {
            "num_foci": n_foci,
            "depth_tier": "mouse_ct_artificial_prior",
            "depth_mm": None,
            "source_type": "gaussian",
            "foci": foci,
            "organ_constraint_passed": True,
            "gt_status": "simulated_training_ground_truth",
            "subject_id": args.subject_id,
        }
        (sample_dir / "tumor_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        np.save(sample_dir / "gt_voxels.npy", evaluate_gaussian_sources(voxel_points, foci).reshape(labels.shape))
        np.save(sample_dir / "gt_nodes.npy", evaluate_gaussian_sources(mesh_nodes, foci))
        sample_names.append(sample_name)
        print(sample_name, foci)
    validation_count = min(args.validation_count, len(sample_names))
    write_split(split_dir / "train_simulated.txt", sample_names[:-validation_count] if validation_count else sample_names)
    write_split(split_dir / "val_simulated.txt", sample_names[-validation_count:] if validation_count else [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
