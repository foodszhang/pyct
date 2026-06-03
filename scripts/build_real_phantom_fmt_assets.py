#!/usr/bin/env python3
"""Build FMT-SimGen shared assets for an imported real phantom bundle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    """Run one FMT-SimGen asset build step."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--fmt-root", type=Path, default=Path("/home/foods/pro/FMT-SimGen"))
    parser.add_argument("--mesh-downsample", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """Generate MCX volume, coarse FEM mesh, and seven-view camera assets."""
    args = parse_args()
    fmt_root = args.fmt_root.resolve()
    python = str(fmt_root / ".venv" / "bin" / "python")
    subject_dir = args.bundle_root.resolve() / "subjects" / args.subject_id
    shared_dir = subject_dir / "shared"
    config = subject_dir / "fmt_simgen_subject.yaml"
    atlas = shared_dir / "atlas_labels.npz"
    labels = shared_dir / "body_labels.nii.gz"
    mesh_prefix = f"{args.subject_id}_mesh"

    step0f = [
        python,
        "scripts/step0f_mcx_volume.py",
        "--config",
        str(config),
        "--atlas_path",
        str(atlas),
        "--output_dir",
        str(shared_dir),
        "--skip_stats",
    ]
    run(step0f, fmt_root)
    run(
        [
            python,
            "scripts/step0b_generate_mesh_cgalmesh.py",
            "--config",
            str(config),
            "--atlas",
            str(labels),
            "--downsample",
            str(args.mesh_downsample),
            "--output-name",
            mesh_prefix,
            "--output-dir",
            str(shared_dir),
            "--no-viz",
        ],
        fmt_root,
    )
    shutil.copy2(shared_dir / f"{mesh_prefix}.npz", shared_dir / "mesh.npz")
    shutil.copy2(shared_dir / f"{mesh_prefix}.mat", shared_dir / "mesh.mat")

    # step0b records the coarse FEM grid in the manifest. MCX keeps the full CT grid.
    run(step0f, fmt_root)
    run(
        [
            python,
            "scripts/step0g_view_config.py",
            "--config",
            str(config),
            "--mesh",
            str(shared_dir / "mesh.npz"),
            "--output-dir",
            str(shared_dir),
        ],
        fmt_root,
    )
    print(f"Wrote FMT-SimGen shared assets: {shared_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
