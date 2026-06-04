#!/usr/bin/env python3
"""Run controlled CBCT preprocessing and filter comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECON_SCRIPT = PROJECT_ROOT / "scripts" / "reconstruct_ct_dir.py"


def filter_slug(name: str) -> str:
    return name.lower().replace("-", "")


def experiment_matrix(full: bool) -> list[dict]:
    base = [
        {"name": "baseline_hamming", "filter": "Hamming"},
        {"name": "gaussian0p8_hamming", "filter": "Hamming", "projection_gaussian_sigma": 0.8},
        {"name": "median3_hamming", "filter": "Hamming", "projection_median_kernel": 3},
        {"name": "filter_hann", "filter": "Hann"},
        {"name": "filter_shepplogan", "filter": "Shepp-Logan"},
        {"name": "filter_ramlak", "filter": "Ram-Lak"},
    ]
    if not full:
        return base
    return base + [
        {"name": "gaussian0p5_hamming", "filter": "Hamming", "projection_gaussian_sigma": 0.5},
        {"name": "gaussian1p2_hamming", "filter": "Hamming", "projection_gaussian_sigma": 1.2},
        {"name": "air99p5_hamming", "filter": "Hamming", "air_percentile": 99.5},
        {"name": "air99p8_hamming", "filter": "Hamming", "air_percentile": 99.8},
        {"name": "clip99p8_hamming", "filter": "Hamming", "intensity_clip_percentile": 99.8},
        {"name": "ring5_hamming", "filter": "Hamming", "ring_kernel_size": 5},
        {"name": "ring15_hamming", "filter": "Hamming", "ring_kernel_size": 15},
    ]


def add_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_command(args: argparse.Namespace, proj_dir: Path, out_dir: Path, exp: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(RECON_SCRIPT),
        str(proj_dir),
        "--calibration",
        str(args.calibration),
        "--config",
        str(args.config),
        "--output-dir",
        str(out_dir),
        "--size",
        str(args.size),
        "--voxel-size",
        str(args.voxel_size),
        "--detector-scale",
        str(args.detector_scale),
        "--filter",
        exp["filter"],
        "--ring-kernel-size",
        str(exp.get("ring_kernel_size", args.ring_kernel_size)),
        "--roi-x",
        str(args.roi_x),
        "--roi-y",
        str(args.roi_y),
        "--roi-z",
        str(args.roi_z),
    ]
    if args.fill_missing_degrees:
        cmd.append("--fill-missing-degrees")
    if args.mass_normalize or exp.get("mass_normalize"):
        cmd.append("--mass-normalize")
    add_optional(cmd, "--air-percentile", exp.get("air_percentile", args.air_percentile))
    add_optional(cmd, "--air-zero-percentile", exp.get("air_zero_percentile", args.air_zero_percentile))
    add_optional(cmd, "--projection-gaussian-sigma", exp.get("projection_gaussian_sigma"))
    add_optional(cmd, "--projection-median-kernel", exp.get("projection_median_kernel"))
    add_optional(cmd, "--intensity-floor", exp.get("intensity_floor", args.intensity_floor))
    add_optional(cmd, "--intensity-clip-percentile", exp.get("intensity_clip_percentile"))
    return cmd


def run_one(args: argparse.Namespace, proj_dir: Path, output_root: Path, exp: dict) -> dict:
    out_dir = output_root / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"rec_vshift_{filter_slug(exp['filter'])}.nii.gz"
    log_path = out_dir / "run.log"
    cmd = build_command(args, proj_dir, out_dir, exp)
    command_record = {"experiment": exp, "projection_dir": str(proj_dir), "command": cmd}
    (out_dir / "experiment_command.json").write_text(
        json.dumps(command_record, indent=2), encoding="utf-8"
    )
    if expected.exists() and not args.overwrite:
        return collect_result(exp["name"], "skipped", expected, out_dir, None)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    status = "done" if proc.returncode == 0 else "failed"
    return collect_result(exp["name"], status, expected, out_dir, proc.returncode, log_path)


def collect_result(
    name: str,
    status: str,
    expected: Path,
    out_dir: Path,
    returncode: int | None,
    log_path: Path | None = None,
) -> dict:
    result = {
        "name": name,
        "status": status,
        "returncode": returncode,
        "output": str(expected),
        "log": str(log_path) if log_path else str(out_dir / "run.log"),
    }
    manifest_path = out_dir / "reconstruction_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        result["filter"] = manifest.get("filter")
        result["preprocess"] = manifest.get("preprocess", {})
        result["quality_metrics"] = manifest.get("quality_metrics", {})
    return result


def write_metrics_csv(results: list[dict], output_root: Path) -> None:
    rows = []
    for result in results:
        metrics = result.get("quality_metrics", {})
        preprocess = result.get("preprocess", {})
        rows.append(
            {
                "name": result["name"],
                "status": result["status"],
                "filter": result.get("filter", ""),
                "gaussian_sigma": preprocess.get("projection_gaussian_sigma", ""),
                "median_kernel": preprocess.get("projection_median_kernel", ""),
                "air_percentile": preprocess.get("air_percentile", ""),
                "clip_percentile": preprocess.get("intensity_clip_percentile", ""),
                "bone_edge_gradient_p95": metrics.get("bone_edge_gradient_p95", ""),
                "soft_local_contrast_p90_p10": metrics.get("soft_local_contrast_p90_p10", ""),
                "soft_std": metrics.get("soft_std", ""),
                "background_std": metrics.get("background_std", ""),
                "output": result.get("output", ""),
            }
        )
    if not rows:
        return
    with open(output_root / "experiment_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-dir", type=Path, default=PROJECT_ROOT / "20260602-jz-2")
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "20260602-jz-2" / "calibration_vshift.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--detector-scale", type=float, default=1.0)
    parser.add_argument("--roi-x", type=float, default=0.0)
    parser.add_argument("--roi-y", type=float, default=0.0)
    parser.add_argument("--roi-z", type=float, default=-60.0)
    parser.add_argument("--ring-kernel-size", type=int, default=9)
    parser.add_argument("--fill-missing-degrees", action="store_true")
    parser.add_argument("--air-percentile", type=float)
    parser.add_argument("--air-zero-percentile", type=float)
    parser.add_argument("--mass-normalize", action="store_true")
    parser.add_argument("--intensity-floor", type=float, default=1.0)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Optional experiment names to run from the selected matrix.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or args.proj_dir / "preprocess_filter_experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    matrix = experiment_matrix(args.full_matrix)
    if args.experiments:
        selected = set(args.experiments)
        matrix = [exp for exp in matrix if exp["name"] in selected]
        missing = selected - {exp["name"] for exp in matrix}
        if missing:
            raise ValueError(f"Unknown experiment(s): {sorted(missing)}")
    for exp in matrix:
        print(f"[Experiment] {exp['name']}")
        result = run_one(args, args.proj_dir, output_root, exp)
        print(f"[{result['status']}] {exp['name']}")
        results.append(result)
    (output_root / "experiment_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_metrics_csv(results, output_root)
    failures = [row for row in results if row["status"] == "failed"]
    if failures:
        print(f"[Failed] {len(failures)} experiment(s)")
        return 1
    print(output_root / "experiment_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
