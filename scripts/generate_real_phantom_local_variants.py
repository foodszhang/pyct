#!/usr/bin/env python3
"""Generate small source-prior perturbations for real-phantom MCX fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Write global-shift and radius hypotheses around the editable baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shift-mm", type=float, default=3.0)
    args = parser.parse_args()
    baseline = json.loads((args.sample_dir / "tumor_params.json").read_text(encoding="utf-8"))
    output_dir = args.sample_dir / "source_local_variants"
    output_dir.mkdir(exist_ok=True)
    variants = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            suffix = f"shift_{'xyz'[axis]}_{'pos' if sign > 0 else 'neg'}{args.shift_mm:g}mm"
            params = json.loads(json.dumps(baseline))
            for focus in params["foci"]:
                focus["center"][axis] = round(float(focus["center"][axis]) + sign * args.shift_mm, 4)
            variants.append((suffix, params))
    for radius in (1.0, 3.0):
        params = json.loads(json.dumps(baseline))
        for focus in params["foci"]:
            focus["radius"] = radius
            focus["params"]["radius"] = radius
        variants.append((f"radius_{radius:g}mm", params))
    for shift_mm in (-6.0, -3.0):
        params = json.loads(json.dumps(baseline))
        for focus in params["foci"]:
            focus["center"][1] = round(float(focus["center"][1]) + shift_mm, 4)
            focus["radius"] = 3.0
            focus["params"]["radius"] = 3.0
        variants.append((f"shift_y_{shift_mm:g}mm_radius_3mm", params))
    for suffix, params in variants:
        params["fit_variant"] = {"type": "local_perturbation", "name": suffix}
        (output_dir / f"{suffix}.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
