#!/usr/bin/env python3
"""Create Digimouse-like CT-supported and approximate smooth tissue labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import filters


LABELS = {
    0: "background",
    1: "soft_tissue",
    2: "bone_ct_supported",
    3: "brain_approx",
    4: "heart_approx",
    5: "stomach_approx",
    6: "abdominal_approx",
    7: "liver_approx",
    8: "kidney_approx",
    9: "lung_approx",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--head-side", choices=["low", "high"], default="high")
    parser.add_argument(
        "--roi-fraction",
        nargs=6,
        type=float,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        default=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        help="Restrict body extraction to fractional volume bounds.",
    )
    parser.add_argument(
        "--smooth-body-prior",
        action="store_true",
        help="Replace rack-contaminated threshold body with a smooth torso/head prior.",
    )
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected 3D component."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def ellipsoid(shape: tuple[int, int, int], center: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return a smooth anatomical template primitive."""
    grid = np.ogrid[tuple(slice(0, size) for size in shape)]
    value = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        value += ((grid[axis] - center[axis]) / max(float(radius[axis]), 1.0)) ** 2
    return value <= 1.0


def template_mask(
    shape: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_size: np.ndarray,
    axis: int,
    head_side: str,
    longitudinal: float,
    transverse_1: float,
    transverse_2: float,
    radius_long: float,
    radius_transverse_1: float,
    radius_transverse_2: float,
) -> np.ndarray:
    """Place an ellipsoid using normalized coordinates inside the body box."""
    other = [index for index in range(3) if index != axis]
    position = longitudinal if head_side == "high" else 1.0 - longitudinal
    center = bbox_min + bbox_size * 0.5
    center[axis] = bbox_min[axis] + bbox_size[axis] * position
    center[other[0]] = bbox_min[other[0]] + bbox_size[other[0]] * transverse_1
    center[other[1]] = bbox_min[other[1]] + bbox_size[other[1]] * transverse_2
    radius = bbox_size * 0.2
    radius[axis] = bbox_size[axis] * radius_long
    radius[other[0]] = bbox_size[other[0]] * radius_transverse_1
    radius[other[1]] = bbox_size[other[1]] * radius_transverse_2
    return ellipsoid(shape, center, radius)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    """Normalize an image for diagnostics."""
    lo, hi = np.percentile(image[np.isfinite(image)], [1.0, 99.5])
    return np.clip((image - lo) / (hi - lo + 1.0e-8) * 255.0, 0, 255).astype(np.uint8)


def save_qc(volume: np.ndarray, labels: np.ndarray, body: np.ndarray, path: Path) -> None:
    """Write central orthogonal label overlays."""
    indices = np.argwhere(body)
    centers = ((indices.min(axis=0) + indices.max(axis=0)) // 2).tolist()
    slices = [
        (volume[:, :, centers[2]], labels[:, :, centers[2]]),
        (volume[:, centers[1], :], labels[:, centers[1], :]),
        (volume[centers[0], :, :], labels[centers[0], :, :]),
    ]
    colors = np.asarray(
        [[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 80, 80], [255, 180, 80],
         [120, 80, 255], [180, 120, 80], [80, 220, 80], [80, 220, 220], [80, 120, 255]],
        dtype=np.uint8,
    )
    panels = []
    for image, tissue in slices:
        base = cv2.cvtColor(normalize_u8(image.T), cv2.COLOR_GRAY2BGR)
        overlay = colors[tissue.T]
        panel = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def save_mip_qc(volume: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Write maximum-intensity projections with projected tissue overlays."""
    colors = np.asarray(
        [[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 80, 80], [255, 180, 80],
         [120, 80, 255], [180, 120, 80], [80, 220, 80], [80, 220, 220], [80, 120, 255]],
        dtype=np.uint8,
    )
    panels = []
    for axis in range(3):
        image = np.max(volume, axis=axis).T
        tissue = np.max(labels, axis=axis).T
        base = cv2.cvtColor(normalize_u8(image), cv2.COLOR_GRAY2BGR)
        overlay = colors[tissue]
        panel = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
        cv2.putText(panel, f"MIP axis={axis}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def main() -> int:
    """Create approximate tissue labels and provenance metadata."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(args.volume))
    volume = image.get_fdata(dtype="float32")
    smooth = ndimage.gaussian_filter(np.nan_to_num(volume), sigma=1.0)
    finite = smooth[np.isfinite(smooth)]
    values = finite[finite > np.percentile(finite, 45.0)]
    threshold = float(filters.threshold_otsu(values))
    roi = np.zeros(volume.shape, dtype=bool)
    bounds = []
    for size, lo, hi in zip(volume.shape, args.roi_fraction[::2], args.roi_fraction[1::2]):
        start = int(np.clip(round(size * lo), 0, size - 1))
        stop = int(np.clip(round(size * hi), start + 1, size))
        bounds.append((start, stop))
    roi[tuple(slice(start, stop) for start, stop in bounds)] = True
    body = largest_component((smooth > threshold) & roi)
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    indices = np.argwhere(body)
    if indices.size == 0:
        raise RuntimeError("Could not determine a body mask")
    bbox_min = indices.min(axis=0).astype(np.float32)
    bbox_size = (indices.max(axis=0) - indices.min(axis=0) + 1).astype(np.float32)
    axis = int(np.argmax(bbox_size)) if args.head_axis == "auto" else {"x": 0, "y": 1, "z": 2}[args.head_axis]
    if args.smooth_body_prior:
        other = [index for index in range(3) if index != axis]
        torso_center = bbox_min + bbox_size * 0.5
        torso_center[axis] = bbox_min[axis] + bbox_size[axis] * (0.55 if args.head_side == "low" else 0.45)
        torso_radius = bbox_size * 0.3
        torso_radius[axis] = bbox_size[axis] * 0.43
        torso_radius[other[0]] = bbox_size[other[0]] * 0.31
        torso_radius[other[1]] = bbox_size[other[1]] * 0.27
        head_center = bbox_min + bbox_size * 0.5
        head_center[axis] = bbox_min[axis] + bbox_size[axis] * (0.10 if args.head_side == "low" else 0.90)
        head_radius = bbox_size * 0.2
        head_radius[axis] = bbox_size[axis] * 0.12
        head_radius[other[0]] = bbox_size[other[0]] * 0.22
        head_radius[other[1]] = bbox_size[other[1]] * 0.22
        body = (ellipsoid(volume.shape, torso_center, torso_radius) | ellipsoid(volume.shape, head_center, head_radius)) & roi
        indices = np.argwhere(body)
        bbox_min = indices.min(axis=0).astype(np.float32)
        bbox_size = (indices.max(axis=0) - indices.min(axis=0) + 1).astype(np.float32)

    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[body] = 1

    def put(label: int, mask: np.ndarray) -> None:
        labels[mask & body] = label

    put(6, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.40, 0.50, 0.50, 0.23, 0.34, 0.34))
    put(3, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.88, 0.50, 0.52, 0.09, 0.24, 0.23))
    put(9, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.67, 0.34, 0.54, 0.13, 0.18, 0.23))
    put(9, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.67, 0.66, 0.54, 0.13, 0.18, 0.23))
    put(4, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.60, 0.50, 0.45, 0.08, 0.15, 0.16))
    put(7, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.49, 0.50, 0.49, 0.10, 0.30, 0.22))
    put(5, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.42, 0.64, 0.52, 0.07, 0.14, 0.15))
    put(8, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.35, 0.34, 0.54, 0.07, 0.11, 0.13))
    put(8, template_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, 0.35, 0.66, 0.54, 0.07, 0.11, 0.13))

    bone_threshold = float(np.percentile(smooth[body], 92.0))
    bone = largest_component((smooth >= bone_threshold) & body)
    bone |= ((smooth >= bone_threshold) & body)
    bone = ndimage.binary_opening(bone, iterations=1)
    labels[bone] = 2

    np.save(args.output_dir / "body_mask.npy", body.astype(np.uint8))
    np.save(args.output_dir / "digimouse_like_labels.npy", labels)
    nib.save(nib.Nifti1Image(labels, image.affine, image.header), args.output_dir / "digimouse_like_labels.nii.gz")
    save_qc(volume, labels, body, args.output_dir / "digimouse_like_labels_qc.png")
    save_mip_qc(volume, labels, args.output_dir / "digimouse_like_labels_mip_qc.png")
    metadata = {
        "version": 1,
        "source_volume": str(args.volume.resolve()),
        "status": "approximate_anatomical_prior_not_manual_segmentation",
        "head_axis": "xyz"[axis],
        "head_side": args.head_side,
        "roi_fraction": args.roi_fraction,
        "smooth_body_prior": args.smooth_body_prior,
        "body_threshold": threshold,
        "bone_threshold": bone_threshold,
        "label_mapping": LABELS,
        "provenance": {
            "body": "ct_supported_largest_component",
            "bone": "ct_supported_high_density",
            "major_organs": "smooth_anatomical_ellipsoid_priors_clipped_to_body",
        },
    }
    (args.output_dir / "segmentation_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_dir / "digimouse_like_labels.nii.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
