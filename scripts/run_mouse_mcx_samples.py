#!/usr/bin/env python3
"""Run Windows MCX for generated mouse simulation samples."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


FMT_SIMGEN_ROOT = Path("/home/foods/pro/FMT-SimGen")
sys.path.insert(0, str(FMT_SIMGEN_ROOT))

from fmt_simgen.mcx_config import generate_mcx_config
from fmt_simgen.mcx_projection import project_sample
from fmt_simgen.mcx_runner import run_mcx_single
from fmt_simgen.view_config import TurntableCamera


def load_json(path: Path) -> dict:
    """Load JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    """Stage Windows-visible inputs and generate seven-view projections."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--mcx-exe", type=Path, default=Path("/mnt/f/win-pro/bin/mcx.exe"))
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/f/win-pro/mcx_mouse_runs"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    shared_dir = args.bundle_root / "subjects" / args.subject_id / "shared"
    samples_dir = args.bundle_root / "samples"
    manifest = load_json(shared_dir / "frame_manifest.json")
    camera = TurntableCamera(load_json(shared_dir / "view_config.json"))
    mcx_volume = manifest["mcx_volume"]
    sample_names = []
    for split_name in ("train_simulated.txt", "val_simulated.txt"):
        sample_names.extend((args.bundle_root / "splits" / split_name).read_text(encoding="utf-8").splitlines())
    for sample_name in sample_names:
        sample_dir = samples_dir / sample_name
        run_dir = args.run_root / args.subject_id / sample_name
        run_dir.mkdir(parents=True, exist_ok=True)
        staged_volume = run_dir / "mcx_volume_trunk.bin"
        if not staged_volume.exists():
            shutil.copy2(shared_dir / "mcx_volume_trunk.bin", staged_volume)
        params = load_json(sample_dir / "tumor_params.json")
        config_path = Path(
            generate_mcx_config(
                sample_name,
                params,
                {
                    "volume_path": str(staged_volume),
                    "material_path": str(shared_dir / "mcx_material.yaml"),
                    "volume_shape": mcx_volume["shape_zyx"],
                    "voxel_size_mm": mcx_volume["voxel_size_mm"],
                },
                run_dir,
            )
        )
        config = load_json(config_path)
        config["Domain"]["VolumeFile"] = staged_volume.name
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        jnii = run_dir / f"{sample_name}.jnii"
        projection = run_dir / "proj.npz"
        if args.force:
            jnii.unlink(missing_ok=True)
            projection.unlink(missing_ok=True)
        run_mcx_single(run_dir, mcx_exec=str(args.mcx_exe), config_name=config_path.name)
        project_sample(
            run_dir,
            camera,
            skip_existing=not args.force,
            voxel_size_mm=float(mcx_volume["voxel_size_mm"]),
            volume_center_world=tuple(manifest["volume_center_world_mm"]),
            jnii_filename=jnii.name,
        )
        shutil.copy2(projection, sample_dir / "proj.npz")
        (sample_dir / "simulation_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "mouse_ct_mcx_simulated_training_pair",
                    "subject_id": args.subject_id,
                    "mcx_run_dir": str(run_dir.resolve()),
                    "projection": "proj.npz",
                    "gt_voxels": "gt_voxels.npy",
                    "gt_nodes": "gt_nodes.npy",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Done] {sample_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
