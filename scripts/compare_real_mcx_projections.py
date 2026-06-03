#!/usr/bin/env python3
"""Compare seven-view MCX candidate projections against processed real fluorescence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


ANGLES = [-90, -60, -30, 0, 30, 60, 90]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    return parser.parse_args()


def normalized(image: np.ndarray, blur_sigma: float) -> np.ndarray:
    """Prepare one projection for shape comparison independent of absolute intensity."""
    image = np.clip(np.nan_to_num(image.astype(np.float32)), 0.0, None)
    if blur_sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=blur_sigma)
    scale = float(np.percentile(image, 99.5))
    if scale <= 1.0e-8:
        return np.zeros_like(image)
    return np.clip(image / scale, 0.0, 1.0)


def metrics(real: np.ndarray, simulated: np.ndarray, blur_sigma: float) -> dict[str, float]:
    """Compute scale-independent projection similarity metrics."""
    real = normalized(real, blur_sigma)
    simulated = normalized(simulated, blur_sigma)
    real_flat = real.ravel().astype(np.float64)
    simulated_flat = simulated.ravel().astype(np.float64)
    real_centered = real_flat - real_flat.mean()
    simulated_centered = simulated_flat - simulated_flat.mean()
    ncc = float(
        np.dot(real_centered, simulated_centered)
        / (np.linalg.norm(real_centered) * np.linalg.norm(simulated_centered) + 1.0e-12)
    )
    cosine = float(
        np.dot(real_flat, simulated_flat)
        / (np.linalg.norm(real_flat) * np.linalg.norm(simulated_flat) + 1.0e-12)
    )
    rmse = float(np.sqrt(np.mean((real_flat - simulated_flat) ** 2)))
    return {"ncc": ncc, "cosine": cosine, "rmse": rmse}


def main() -> int:
    """Write per-view and aggregate MCX fitting metrics."""
    args = parse_args()
    with np.load(args.real) as real_archive, np.load(args.simulated) as sim_archive:
        per_view = {}
        for angle in ANGLES:
            key = str(angle)
            if key not in real_archive.files or key not in sim_archive.files:
                raise KeyError(f"Both archives must contain angle {key}")
            per_view[key] = metrics(real_archive[key], sim_archive[key], args.blur_sigma)
    summary = {
        name: float(np.mean([item[name] for item in per_view.values()]))
        for name in ("ncc", "cosine", "rmse")
    }
    result = {
        "version": 1,
        "real": str(args.real.resolve()),
        "simulated": str(args.simulated.resolve()),
        "blur_sigma": args.blur_sigma,
        "per_view": per_view,
        "mean": summary,
        "objective": "maximize mean.ncc then mean.cosine; minimize mean.rmse",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
