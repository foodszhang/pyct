#!/usr/bin/env python3
"""Rank isolated real-phantom MCX candidate directories by projection similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Write a ranked JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for report_path in args.run_dir.glob("candidate_*/projection_similarity.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        params = json.loads((report_path.parent / "tumor_params.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate_id": report_path.parent.name,
                "fit_variant": params.get("fit_variant", {"type": "baseline"}),
                **report["mean"],
            }
        )
    rows.sort(key=lambda item: (-item["ncc"], -item["cosine"], item["rmse"]))
    result = {"version": 1, "objective": "maximize ncc then cosine; minimize rmse", "ranking": rows}
    output = args.output or args.run_dir / "candidate_ranking.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        print(f"{row['candidate_id']} ncc={row['ncc']:.6f} cosine={row['cosine']:.6f} rmse={row['rmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
