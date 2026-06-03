#!/usr/bin/env python3
"""Rank true_data mouse acquisitions for real FMT dataset preparation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    """Join inventory and CT QC, then write a pragmatic processing priority report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/foods/pro/true_data/ZJ"))
    args = parser.parse_args()
    inventory = {row["subject"]: row for row in read_csv(args.root / "dataset_inventory" / "mouse_dataset_inventory.csv")}
    qc = read_csv(args.root / "cbct_qc" / "summary.csv")
    rows = []
    for item in qc:
        subject = item["sample"]
        flu_frames = int(inventory[subject]["flu_frame_count"])
        clean = int(item["clean_count"])
        missing = int(item["missing_count"])
        suspicious_ratio = float(item["suspicious_ratio"])
        severe_ratio = float(item["severe_ratio"])
        eligible = clean > 0 and flu_frames >= 7
        score = severe_ratio + 0.5 * suspicious_ratio + missing / 360.0
        rows.append(
            {
                "subject": subject,
                "eligible_for_7view_real_test": int(eligible),
                "priority_score_lower_is_better": score,
                "ct_clean_projection_count": clean,
                "ct_missing_angle_count": missing,
                "ct_suspicious_ratio": suspicious_ratio,
                "ct_severe_ratio": severe_ratio,
                "flu_frame_count": flu_frames,
                "reconstruction_count": int(inventory[subject]["reconstruction_count"]),
            }
        )
    rows.sort(key=lambda row: (not row["eligible_for_7view_real_test"], row["priority_score_lower_is_better"]))
    output_dir = args.root / "dataset_inventory"
    with open(output_dir / "mouse_dataset_priority.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "mouse_dataset_priority.json").write_text(
        json.dumps({"version": 1, "ranking": rows}, indent=2), encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['subject']:>2} eligible={row['eligible_for_7view_real_test']} "
            f"score={row['priority_score_lower_is_better']:.4f} "
            f"flu={row['flu_frame_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
