#!/usr/bin/env python3
"""Split one multi-focus source prior into single-focus MCX diagnostic variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Write one MCX source-parameter file per focus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.tumor_params.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, focus in enumerate(baseline["foci"], 1):
        params = json.loads(json.dumps(baseline))
        params["num_foci"] = 1
        params["foci"] = [focus]
        params["fit_variant"] = {
            "type": "single_focus_diagnostic",
            "focus_index": index,
            "focus_name": focus.get("name", f"source_{index}"),
        }
        path = args.output_dir / f"focus_{index}.json"
        path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
