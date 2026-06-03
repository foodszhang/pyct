#!/usr/bin/env python3
"""Generate coordinate-frame source variants for initial real-phantom fitting."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path


def main() -> int:
    """Write reflected source hypotheses without changing the editable baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, required=True)
    args = parser.parse_args()
    params = json.loads((args.sample_dir / "tumor_params.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.shared_dir / "frame_manifest.json").read_text(encoding="utf-8"))
    extent = manifest["mcx_volume"]["extent_mm"]
    output_dir = args.sample_dir / "source_variants"
    output_dir.mkdir(exist_ok=True)
    for flip_x, flip_y, flip_z in product((False, True), repeat=3):
        variant = json.loads(json.dumps(params))
        suffix = "".join(axis for axis, enabled in zip("xyz", (flip_x, flip_y, flip_z)) if enabled) or "identity"
        for focus in variant["foci"]:
            center = focus["center"]
            for axis, enabled in enumerate((flip_x, flip_y, flip_z)):
                if enabled:
                    center[axis] = round(float(extent[axis]) - float(center[axis]), 4)
        variant["fit_variant"] = {"type": "coordinate_reflection", "axes": suffix}
        (output_dir / f"reflection_{suffix}.json").write_text(
            json.dumps(variant, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
