#!/usr/bin/env python3
"""Generate tissue-constrained local source variants for real mouse fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def safe_number(value: float) -> str:
    """Encode a signed decimal value for a portable filename."""
    return f"{value:+g}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def main() -> int:
    """Write local center and radius hypotheses around one fitted mouse source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--shift-mm", type=float, nargs="+", default=[0.75, 1.5])
    parser.add_argument("--radius-mm", type=float, nargs="+", default=[1.0, 1.5, 2.5, 3.0])
    parser.add_argument("--allowed-labels", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()

    baseline_path = args.baseline or args.sample_dir / "tumor_params_fitted_estimated.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if len(baseline["foci"]) != 1:
        raise ValueError("Mouse local fitting currently expects exactly one focus")

    labels = np.asarray(nib.load(str(args.shared_dir / "body_labels.nii.gz")).dataobj, dtype=np.uint8)
    manifest = json.loads((args.shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    spacing = float(manifest["mcx_volume"]["voxel_size_mm"])
    output_dir = args.sample_dir / "real_source_local_variants"
    output_dir.mkdir(exist_ok=True)

    variants: list[tuple[str, dict]] = []
    for axis, axis_name in enumerate("xyz"):
        for distance in args.shift_mm:
            for sign in (-1.0, 1.0):
                params = json.loads(json.dumps(baseline))
                center = np.asarray(params["foci"][0]["center"], dtype=np.float64)
                center[axis] += sign * distance
                voxel = np.floor(center / spacing).astype(int)
                if np.any(voxel < 0) or np.any(voxel >= np.asarray(labels.shape)):
                    continue
                tissue_label = int(labels[tuple(voxel)])
                if tissue_label not in args.allowed_labels:
                    continue
                params["foci"][0]["center"] = [round(float(value), 4) for value in center]
                params["foci"][0]["params"]["placement_tissue_label"] = tissue_label
                shift_name = f"shift_{axis_name}_{safe_number(sign * distance)}mm"
                variants.append((shift_name, params))
                for radius in args.radius_mm:
                    combined = json.loads(json.dumps(params))
                    combined["foci"][0]["radius"] = radius
                    combined["foci"][0]["params"]["radius"] = radius
                    variants.append((f"{shift_name}_radius_{safe_number(radius)}mm", combined))

    for radius in args.radius_mm:
        params = json.loads(json.dumps(baseline))
        params["foci"][0]["radius"] = radius
        params["foci"][0]["params"]["radius"] = radius
        variants.append((f"radius_{safe_number(radius)}mm", params))

    summary = []
    for name, params in variants:
        params["gt_status"] = "local_prior_requires_mcx_projection_fit"
        params["fit_variant"] = {"type": "mouse_real_local_perturbation", "name": name}
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        summary.append({"name": name, "path": str(path), "focus": params["foci"][0]})
    (output_dir / "variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "variant_count": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
