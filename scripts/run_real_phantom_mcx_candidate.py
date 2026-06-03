#!/usr/bin/env python3
"""Run one estimated real-phantom source candidate through Windows MCX."""

from __future__ import annotations

import argparse
import hashlib
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--tumor-params", type=Path)
    parser.add_argument("--mcx-exe", type=Path, default=Path("/mnt/f/win-pro/bin/mcx.exe"))
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/f/win-pro/mcx_runs"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Write stable human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def candidate_hash(tumor_params: dict) -> str:
    """Return a short stable identifier for one adjustable source hypothesis."""
    canonical = json.dumps(tumor_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    """Stage Windows-visible files, run MCX, project fluence, and score the result."""
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    shared_dir = args.shared_dir.resolve()
    tumor_params_path = args.tumor_params or sample_dir / "tumor_params.json"
    tumor_params = load_json(tumor_params_path.resolve())
    manifest = load_json(shared_dir / "frame_manifest.json")
    view_config = load_json(shared_dir / "view_config.json")
    digest = candidate_hash(tumor_params)
    candidate_id = f"candidate_{digest}"
    candidate_dir = args.run_root.resolve() / sample_dir.name / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    jnii_path = candidate_dir / f"{candidate_id}.jnii"
    projection_path = candidate_dir / "proj_mcx_candidate.npz"
    if args.force:
        jnii_path.unlink(missing_ok=True)
        projection_path.unlink(missing_ok=True)

    # Windows MCX can reliably read files staged on /mnt/f. Keeping VolumeFile
    # relative also makes the candidate directory portable on the Windows side.
    staged_volume = candidate_dir / "mcx_volume_trunk.bin"
    if not staged_volume.exists() or staged_volume.stat().st_size == 0:
        shutil.copy2(shared_dir / "mcx_volume_trunk.bin", staged_volume)

    mcx_volume = manifest["mcx_volume"]
    mcx_config = {
        "volume_path": str(staged_volume),
        "material_path": str(shared_dir / "mcx_material.yaml"),
        "volume_shape": mcx_volume["shape_zyx"],
        "voxel_size_mm": mcx_volume["voxel_size_mm"],
    }
    config_path = Path(generate_mcx_config(candidate_id, tumor_params, mcx_config, candidate_dir))
    config = load_json(config_path)
    config["Domain"]["VolumeFile"] = staged_volume.name
    write_json(config_path, config)
    write_json(candidate_dir / "tumor_params.json", tumor_params)

    run_mcx_single(candidate_dir, mcx_exec=str(args.mcx_exe), config_name=config_path.name)
    camera = TurntableCamera(view_config)
    project_sample(
        candidate_dir,
        camera,
        skip_existing=not args.force,
        voxel_size_mm=float(mcx_volume["voxel_size_mm"]),
        volume_center_world=tuple(float(value) for value in manifest["volume_center_world_mm"]),
        jnii_filename=jnii_path.name,
        output_filename=projection_path.name,
    )

    compare_script = Path(__file__).with_name("compare_real_mcx_projections.py")
    report_path = candidate_dir / "projection_similarity.json"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--real",
            str(sample_dir / "proj.npz"),
            "--simulated",
            str(projection_path),
            "--output",
            str(report_path),
        ],
        check=True,
    )
    summary = load_json(report_path)["mean"]
    write_json(
        candidate_dir / "candidate_manifest.json",
        {
            "version": 1,
            "candidate_id": candidate_id,
            "source_params": str((candidate_dir / "tumor_params.json").resolve()),
            "mcx_config": str(config_path.resolve()),
            "fluence": str(jnii_path.resolve()),
            "projection": str(projection_path.resolve()),
            "similarity_report": str(report_path.resolve()),
            "mean_similarity": summary,
        },
    )
    print(json.dumps({"candidate_dir": str(candidate_dir), "mean": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
