#!/usr/bin/env python3
"""Write agar-water optical material parameters for the real phantom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def agar_material_list(
    mua_mm: float,
    musp_mm: float,
    g: float,
    n: float,
) -> list[dict[str, float | int | str]]:
    """Build an MCX media list for a binary agar phantom."""
    mus_mm = musp_mm / (1.0 - g) if musp_mm > 0 and g < 1.0 else 0.0
    return [
        {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0, "tag": 0, "name": "background"},
        {
            "mua": float(mua_mm),
            "mus": float(mus_mm),
            "g": float(g),
            "n": float(n),
            "tag": 1,
            "name": "agar_water_2p22pct_wv",
        },
    ]


def main() -> int:
    """Write material YAML and a provenance sidecar."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mua-mm", type=float, default=0.002)
    parser.add_argument("--musp-mm", type=float, default=0.06)
    parser.add_argument("--g", type=float, default=0.90)
    parser.add_argument("--n", type=float, default=1.334)
    args = parser.parse_args()

    materials = agar_material_list(args.mua_mm, args.musp_mm, args.g, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    provenance = {
        "version": 1,
        "recipe": "45 ml water + 1 g agar powder",
        "agar_concentration_wv_percent": 100.0 / 45.0,
        "assumption": "near-NIR agar-water gel; no added absorber/scatterer recorded",
        "units": "mm^-1",
        "mua_mm": args.mua_mm,
        "musp_mm": args.musp_mm,
        "mus_mm": materials[1]["mus"],
        "g": args.g,
        "n": args.n,
        "note": "MCX uses mus, while this estimate is specified as reduced scattering musp.",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
