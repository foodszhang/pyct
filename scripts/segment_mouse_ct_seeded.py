#!/usr/bin/env python3
"""Segment a mouse CT using CT-supported seeds and ROI constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage


LABELS = {
    0: "background",
    1: "soft_tissue_ct_supported_body",
    2: "bone_ct_supported",
    3: "brain_approx_inside_ct_body",
    4: "heart_approx_inside_ct_body",
    5: "stomach_approx_inside_ct_body",
    6: "abdominal_approx_inside_ct_body",
    7: "liver_approx_inside_ct_body",
    8: "kidney_approx_inside_ct_body",
    9: "lung_approx_inside_ct_body",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-fraction", nargs=6, type=float, default=[0.06, 0.94, 0.58, 0.95, 0.21, 0.73])
    parser.add_argument("--body-percentile", type=float, default=88.0)
    parser.add_argument("--bone-percentile", type=float, default=98.8)
    parser.add_argument("--core-open-iterations", type=int, default=0)
    parser.add_argument("--bone-dilate-iterations", type=int, default=1)
    parser.add_argument("--irregular-organs", action="store_true")
    parser.add_argument("--organ-irregular-strength", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margin-mm", type=float, default=4.0)
    parser.add_argument("--head-side", choices=["low", "high"], default="high")
    return parser.parse_args()


def normalize_u8(image: np.ndarray) -> np.ndarray:
    """Normalize one diagnostic image."""
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.7])
    return np.clip((image - lo) / (hi - lo + 1.0e-8) * 255.0, 0, 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def roi_mask(shape: tuple[int, int, int], fractions: list[float]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Create a 3D ROI mask from fractional bounds."""
    roi = np.zeros(shape, dtype=bool)
    bounds = []
    for size, lo, hi in zip(shape, fractions[::2], fractions[1::2]):
        start = int(np.clip(round(size * lo), 0, size - 1))
        stop = int(np.clip(round(size * hi), start + 1, size))
        bounds.append((start, stop))
    roi[tuple(slice(start, stop) for start, stop in bounds)] = True
    return roi, bounds


def ellipsoid(shape: tuple[int, int, int], center: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Create an ellipsoid mask."""
    grid = np.ogrid[tuple(slice(0, size) for size in shape)]
    value = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        value += ((grid[axis] - center[axis]) / max(float(radius[axis]), 1.0)) ** 2
    return value <= 1.0


def irregularize(mask: np.ndarray, seed: int, strength: float = 0.18, sigma: float = 7.0) -> np.ndarray:
    """Perturb a smooth organ mask with deterministic low-frequency noise."""
    if not np.any(mask):
        return mask
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 8, 0)
    hi = np.minimum(coords.max(axis=0) + 9, mask.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = mask[slices]
    noise = rng.normal(0.0, 1.0, size=local.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=sigma)
    noise = (noise - noise.mean()) / (noise.std() + 1.0e-6)
    signed = ndimage.distance_transform_edt(local) - ndimage.distance_transform_edt(~local)
    perturbed = signed + noise * strength * max(local.shape)
    output = np.zeros(mask.shape, dtype=bool)
    output[slices] = perturbed > 0
    output = ndimage.binary_opening(output, iterations=1)
    output = ndimage.binary_closing(output, iterations=1)
    return output


def put_template(
    labels: np.ndarray,
    body: np.ndarray,
    bbox_min: np.ndarray,
    bbox_size: np.ndarray,
    label: int,
    longitudinal: float,
    lateral: float,
    depth: float,
    radius_long: float,
    radius_lateral: float,
    radius_depth: float,
    head_side: str,
    irregular: bool,
    seed: int,
    irregular_strength: float,
) -> None:
    """Place one organ prior inside the CT-supported body mask."""
    pos = longitudinal if head_side == "high" else 1.0 - longitudinal
    center = bbox_min + bbox_size * np.asarray([pos, lateral, depth], dtype=np.float32)
    radius = bbox_size * np.asarray([radius_long, radius_lateral, radius_depth], dtype=np.float32)
    organ = ellipsoid(labels.shape, center, radius)
    if irregular:
        organ = irregularize(organ, seed=seed, strength=irregular_strength)
    labels[organ & body] = label


def crop_to_body(volume: np.ndarray, labels: np.ndarray, body: np.ndarray, spacing: float, margin_mm: float) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Crop arrays to body bbox plus margin."""
    pts = np.argwhere(body)
    margin = int(np.ceil(margin_mm / spacing))
    lo = np.maximum(pts.min(axis=0) - margin, 0)
    hi = np.minimum(pts.max(axis=0) + margin + 1, volume.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return volume[slices], labels[slices], [[int(a), int(b)] for a, b in zip(lo, hi)]


def save_overlay_mip(volume: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Save MIP label overlays."""
    colors = np.asarray(
        [[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 80, 80], [255, 180, 80],
         [120, 80, 255], [180, 120, 80], [80, 220, 80], [80, 220, 220], [80, 120, 255]],
        dtype=np.uint8,
    )
    panels = []
    priority = np.asarray([0, 1, 99, 50, 40, 30, 20, 45, 35, 25], dtype=np.uint8)
    for axis in range(3):
        base = cv2.cvtColor(normalize_u8(np.max(volume, axis=axis).T), cv2.COLOR_GRAY2BGR)
        pr = np.max(priority[labels], axis=axis).T
        tissue = np.zeros(pr.shape, dtype=np.uint8)
        for label, rank in enumerate(priority):
            tissue[pr == rank] = label
        panel = cv2.addWeighted(base, 0.58, colors[tissue], 0.42, 0)
        cv2.putText(panel, f"MIP axis={axis}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.hstack(panels))


def save_center_slices(volume: np.ndarray, labels: np.ndarray, body: np.ndarray, path: Path) -> None:
    """Save center slice overlays through body bbox."""
    pts = np.argwhere(body)
    center = ((pts.min(axis=0) + pts.max(axis=0)) // 2).astype(int)
    colors = np.asarray(
        [[0, 0, 0], [90, 90, 90], [255, 255, 255], [255, 80, 80], [255, 180, 80],
         [120, 80, 255], [180, 120, 80], [80, 220, 80], [80, 220, 220], [80, 120, 255]],
        dtype=np.uint8,
    )
    slices = [(volume[:, :, center[2]], labels[:, :, center[2]]), (volume[:, center[1], :], labels[:, center[1], :]), (volume[center[0], :, :], labels[center[0], :, :])]
    panels = []
    for image, tissue in slices:
        base = cv2.cvtColor(normalize_u8(image.T), cv2.COLOR_GRAY2BGR)
        panel = cv2.addWeighted(base, 0.58, colors[tissue.T], 0.42, 0)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.hstack(panels))


def main() -> int:
    """Generate CT-seeded segmentation and cropped canonical volume."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(args.volume))
    volume = image.get_fdata(dtype="float32")
    spacing = float(image.header.get_zooms()[0])
    smooth = ndimage.gaussian_filter(np.nan_to_num(volume), sigma=1.0)
    roi, bounds = roi_mask(volume.shape, args.roi_fraction)
    body_threshold = float(np.percentile(smooth[roi], args.body_percentile))
    candidate = (smooth > body_threshold) & roi
    body = largest_component(candidate)
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_opening(body, iterations=1)
    body = largest_component(body)
    if args.core_open_iterations > 0:
        core = ndimage.binary_opening(body, iterations=args.core_open_iterations)
        core = largest_component(core)
        core = ndimage.binary_closing(core, iterations=max(1, args.core_open_iterations // 2))
        core = ndimage.binary_fill_holes(core)
        if np.count_nonzero(core) > 0:
            body = core

    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[body] = 1
    bone_threshold = float(np.percentile(smooth[body], args.bone_percentile))
    bone = (smooth >= bone_threshold) & body
    bone = ndimage.binary_opening(bone, iterations=1)
    if args.bone_dilate_iterations > 0:
        bone = ndimage.binary_dilation(bone, iterations=args.bone_dilate_iterations) & body
    labels[bone] = 2

    pts = np.argwhere(body)
    bbox_min = pts.min(axis=0).astype(np.float32)
    bbox_size = (pts.max(axis=0) - pts.min(axis=0) + 1).astype(np.float32)
    put_template(labels, body, bbox_min, bbox_size, 6, 0.40, 0.50, 0.50, 0.23, 0.34, 0.34, args.head_side, args.irregular_organs, args.seed + 6, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 3, 0.88, 0.50, 0.52, 0.09, 0.22, 0.22, args.head_side, args.irregular_organs, args.seed + 3, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 9, 0.66, 0.36, 0.54, 0.13, 0.16, 0.22, args.head_side, args.irregular_organs, args.seed + 91, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 9, 0.66, 0.64, 0.54, 0.13, 0.16, 0.22, args.head_side, args.irregular_organs, args.seed + 92, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 4, 0.58, 0.50, 0.47, 0.08, 0.14, 0.14, args.head_side, args.irregular_organs, args.seed + 4, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 7, 0.48, 0.49, 0.50, 0.10, 0.28, 0.20, args.head_side, args.irregular_organs, args.seed + 7, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 5, 0.39, 0.63, 0.53, 0.07, 0.13, 0.13, args.head_side, args.irregular_organs, args.seed + 5, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 8, 0.34, 0.35, 0.54, 0.07, 0.10, 0.12, args.head_side, args.irregular_organs, args.seed + 81, args.organ_irregular_strength)
    put_template(labels, body, bbox_min, bbox_size, 8, 0.34, 0.65, 0.54, 0.07, 0.10, 0.12, args.head_side, args.irregular_organs, args.seed + 82, args.organ_irregular_strength)
    labels[bone] = 2

    body_only_volume = np.where(body, volume, 0.0).astype(np.float32)
    cropped_volume, cropped_labels, crop_bbox = crop_to_body(body_only_volume, labels, body, spacing, args.margin_mm)
    cropped_body = cropped_labels > 0
    affine = image.affine.copy()
    affine[:3, 3] += np.asarray([crop_bbox[0][0], crop_bbox[1][0], crop_bbox[2][0]], dtype=np.float64) * spacing
    nib.save(nib.Nifti1Image(labels, image.affine, image.header), args.output_dir / "digimouse_like_labels.nii.gz")
    nib.save(nib.Nifti1Image(body_only_volume, image.affine), args.output_dir / "ct_bodyonly.nii.gz")
    nib.save(nib.Nifti1Image(cropped_volume.astype(np.float32), affine), args.output_dir / "ct_cropped.nii.gz")
    nib.save(nib.Nifti1Image(cropped_labels.astype(np.uint8), affine), args.output_dir / "digimouse_like_labels_cropped.nii.gz")
    np.save(args.output_dir / "body_mask.npy", body.astype(np.uint8))
    save_overlay_mip(volume, labels, args.output_dir / "digimouse_like_labels_mip_qc.png")
    save_center_slices(volume, labels, body, args.output_dir / "digimouse_like_labels_qc.png")
    save_overlay_mip(cropped_volume, cropped_labels, args.output_dir / "cropped_labels_mip_qc.png")
    save_center_slices(cropped_volume, cropped_labels, cropped_body, args.output_dir / "cropped_labels_qc.png")

    manifest = {
        "version": 1,
        "source_volume": str(args.volume.resolve()),
        "status": "ct_seeded_body_and_bone_with_anatomical_organ_priors",
        "roi_fraction": args.roi_fraction,
        "roi_bounds_voxel": bounds,
        "body_threshold": body_threshold,
        "bone_threshold": bone_threshold,
        "core_open_iterations": args.core_open_iterations,
        "bone_dilate_iterations": args.bone_dilate_iterations,
        "irregular_organs": args.irregular_organs,
        "organ_irregular_strength": args.organ_irregular_strength,
        "seed": args.seed,
        "crop_bbox_voxel_xyz": crop_bbox,
        "voxel_size_mm": spacing,
        "label_mapping": LABELS,
        "provenance": {
            "body": "largest_component_inside_user_roi_after_ct_intensity_threshold",
            "bone": "high_density_ct_supported_inside_body",
            "major_organs": "anatomical_priors_clipped_to_ct_supported_body",
            "rack_removal": "exclude large rack slabs using ROI before largest-component selection",
        },
    }
    (args.output_dir / "segmentation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_dir / "digimouse_like_labels_cropped.nii.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
