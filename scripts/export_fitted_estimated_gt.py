#!/usr/bin/env python3
"""Export fitted estimated voxel and mesh-node GT without claiming measured truth."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from compare_real_mcx_projections import ANGLES, normalized
from estimate_real_phantom_gt import evaluate_gaussian_sources


def main() -> int:
    """Export fitted source arrays and provenance metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    args = parser.parse_args()
    params = json.loads((args.candidate_dir / "tumor_params.json").read_text(encoding="utf-8"))
    similarity = json.loads((args.candidate_dir / "projection_similarity.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    shape_xyz = tuple(int(value) for value in manifest["mcx_volume"]["shape_xyz"])
    spacing = float(manifest["mcx_volume"]["voxel_size_mm"])
    grid = np.stack(
        np.meshgrid(
            np.arange(shape_xyz[0], dtype=np.float32),
            np.arange(shape_xyz[1], dtype=np.float32),
            np.arange(shape_xyz[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    points_mm = (grid.reshape(-1, 3) + 0.5) * spacing
    voxels = evaluate_gaussian_sources(points_mm, params["foci"]).reshape(shape_xyz)
    mesh = np.load(args.shared_dir / "mesh.npz")
    nodes = evaluate_gaussian_sources(mesh["nodes"].astype(np.float32), params["foci"])
    np.save(args.sample_dir / "gt_voxels_fitted_estimated.npy", voxels.astype(np.float32))
    np.save(args.sample_dir / "gt_nodes_fitted_estimated.npy", nodes.astype(np.float32))
    shutil.copy2(args.candidate_dir / "tumor_params.json", args.sample_dir / "tumor_params_fitted_estimated.json")
    shutil.copy2(args.candidate_dir / "proj_mcx_candidate.npz", args.sample_dir / "proj_mcx_fitted_estimated.npz")
    shutil.copy2(args.candidate_dir / "projection_similarity.json", args.sample_dir / "projection_similarity_fitted_estimated.json")
    with np.load(args.sample_dir / "proj.npz") as real, np.load(args.candidate_dir / "proj_mcx_candidate.npz") as simulated:
        rows = []
        for angle in ANGLES:
            real_u8 = (normalized(real[str(angle)], 2.0) * 255.0).astype(np.uint8)
            sim_u8 = (normalized(simulated[str(angle)], 2.0) * 255.0).astype(np.uint8)
            tile = np.hstack([real_u8, sim_u8])
            cv2.putText(tile, f"{angle:+d} real | mcx", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            rows.append(tile)
        cv2.imwrite(str(args.sample_dir / "proj_real_vs_mcx_fitted_estimated.png"), np.vstack(rows))
    result = {
        "version": 1,
        "status": "fitted_estimated_prior_not_measured_ground_truth",
        "candidate_dir": str(args.candidate_dir.resolve()),
        "mean_projection_similarity": similarity["mean"],
        "voxel_gt": "gt_voxels_fitted_estimated.npy",
        "node_gt": "gt_nodes_fitted_estimated.npy",
        "tumor_params": "tumor_params_fitted_estimated.json",
        "projection": "proj_mcx_fitted_estimated.npz",
        "projection_comparison": "proj_real_vs_mcx_fitted_estimated.png",
    }
    (args.sample_dir / "fitted_estimated_gt_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.sample_dir / "fitted_estimated_gt_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
