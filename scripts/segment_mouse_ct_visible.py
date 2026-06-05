#!/usr/bin/env python3
"""Conservative mouse CT segmentation from visible structures.

This script is intentionally less atlas-like than ``segment_mouse_ct_seeded.py``.
It keeps bone CT-threshold supported, extracts a mouse body around the skeleton,
and only adds coarse visible regions that are useful for optical simulation.
"""

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
    1: "soft_tissue_visible_body",
    2: "bone_high_density_ct_supported",
    3: "brain_region_simulated_optional",
    4: "heart_region_simulated",
    5: "stomach_region_simulated",
    7: "abdominal_solid_tissue_region",
    8: "kidney_region_simulated",
    9: "lung_low_density_ct_supported",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--head-side", choices=["low", "high"], default="low")
    parser.add_argument("--body-percentile", type=float, default=67.0)
    parser.add_argument("--bone-percentile", type=float, default=98.55)
    parser.add_argument("--lung-percentile", type=float, default=35.0)
    parser.add_argument("--skeleton-dilate", type=int, default=10)
    parser.add_argument("--body-close", type=int, default=4)
    parser.add_argument("--crop-margin-mm", type=float, default=4.0)
    parser.add_argument("--min-component-voxels", type=int, default=2000)
    parser.add_argument(
        "--simulate-major-organs",
        action="store_true",
        help="Place coarse heart/liver/kidney/stomach labels by anatomy, clipped to body and excluding bone.",
    )
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Return largest connected component."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def components_touching_seed(mask: np.ndarray, seed: np.ndarray, min_size: int) -> np.ndarray:
    """Keep components that touch the seed and are not tiny."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    touches = np.bincount(labels[seed & mask].ravel(), minlength=len(sizes)) > 0
    keep = (sizes >= min_size) & touches
    keep[0] = False
    return keep[labels]


def largest_n_components(mask: np.ndarray, n_components: int, min_size: int) -> np.ndarray:
    """Keep the largest N components after removing tiny fragments."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    keep_ids = [int(label) for label in order[:n_components] if sizes[label] >= min_size]
    if not keep_ids:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(labels, keep_ids)


def fractional_slab(shape: tuple[int, int, int], axis: int, low: float, high: float) -> np.ndarray:
    """Create a longitudinal slab mask."""
    mask = np.zeros(shape, dtype=bool)
    start = int(np.clip(round(shape[axis] * low), 0, shape[axis] - 1))
    stop = int(np.clip(round(shape[axis] * high), start + 1, shape[axis]))
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(start, stop)
    mask[tuple(slices)] = True
    return mask


def ellipsoid(shape: tuple[int, int, int], center: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Return an ellipsoid mask."""
    grid = np.ogrid[tuple(slice(0, size) for size in shape)]
    value = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        value += ((grid[axis] - center[axis]) / max(float(radius[axis]), 1.0)) ** 2
    return value <= 1.0


def normalized_center_radius(
    shape: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_size: np.ndarray,
    long_axis: int,
    head_side: str,
    longitudinal: float,
    lateral: float,
    depth: float,
    radius_long: float,
    radius_lateral: float,
    radius_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center/radius in voxel coordinates from body-bbox normalized coordinates."""
    other = [index for index in range(3) if index != long_axis]
    pos = longitudinal if head_side == "low" else 1.0 - longitudinal
    center = bbox_min.astype(np.float32) + bbox_size.astype(np.float32) * 0.5
    center[long_axis] = bbox_min[long_axis] + bbox_size[long_axis] * pos
    center[other[0]] = bbox_min[other[0]] + bbox_size[other[0]] * lateral
    center[other[1]] = bbox_min[other[1]] + bbox_size[other[1]] * depth
    radius = np.maximum(bbox_size.astype(np.float32) * 0.08, 2.0)
    radius[long_axis] = max(float(bbox_size[long_axis] * radius_long), 2.0)
    radius[other[0]] = max(float(bbox_size[other[0]] * radius_lateral), 2.0)
    radius[other[1]] = max(float(bbox_size[other[1]] * radius_depth), 2.0)
    return center, radius


def organ_ellipsoid(
    shape: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_size: np.ndarray,
    long_axis: int,
    head_side: str,
    longitudinal: float,
    lateral: float,
    depth: float,
    radius_long: float,
    radius_lateral: float,
    radius_depth: float,
) -> np.ndarray:
    """Place one organ ellipsoid in body-bbox normalized coordinates."""
    center, radius = normalized_center_radius(
        shape,
        bbox_min,
        bbox_size,
        long_axis,
        head_side,
        longitudinal,
        lateral,
        depth,
        radius_long,
        radius_lateral,
        radius_depth,
    )
    return ellipsoid(shape, center, radius)


def irregularize(mask: np.ndarray, seed: int, strength: float = 0.075, sigma: float = 8.0) -> np.ndarray:
    """Make a smooth template less spherical with deterministic low-frequency perturbation."""
    if not np.any(mask):
        return mask
    rng = np.random.default_rng(seed)
    points = np.argwhere(mask)
    lo = np.maximum(points.min(axis=0) - 8, 0)
    hi = np.minimum(points.max(axis=0) + 9, mask.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = mask[slices]
    noise = rng.normal(0.0, 1.0, size=local.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=sigma)
    noise = (noise - noise.mean()) / (noise.std() + 1.0e-6)
    signed = ndimage.distance_transform_edt(local) - ndimage.distance_transform_edt(~local)
    perturbed = signed + noise * strength * max(local.shape)
    out = np.zeros(mask.shape, dtype=bool)
    out[slices] = perturbed > 0
    return ndimage.binary_closing(ndimage.binary_opening(out, iterations=1), iterations=1)


def digimouse_like_organ_mask(
    shape: tuple[int, int, int],
    bbox_min: np.ndarray,
    bbox_size: np.ndarray,
    long_axis: int,
    head_side: str,
    parts: list[tuple[float, float, float, float, float, float]],
    subtractors: list[tuple[float, float, float, float, float, float]],
    seed: int,
) -> np.ndarray:
    """Build an irregular organ from multiple ellipsoids and small carved notches."""
    mask = np.zeros(shape, dtype=bool)
    for item in parts:
        mask |= organ_ellipsoid(shape, bbox_min, bbox_size, long_axis, head_side, *item)
    for item in subtractors:
        mask &= ~organ_ellipsoid(shape, bbox_min, bbox_size, long_axis, head_side, *item)
    return irregularize(mask, seed=seed)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    """Normalize for PNG QC."""
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1.0, 99.7])
    return np.clip((image - lo) / (hi - lo + 1.0e-8) * 255.0, 0, 255).astype(np.uint8)


def label_projection(labels: np.ndarray, axis: int) -> np.ndarray:
    """Project labels by priority rather than raw max label."""
    priority = np.zeros(10, dtype=np.uint8)
    priority[1] = 10
    priority[3] = 25
    priority[4] = 35
    priority[5] = 32
    priority[7] = 30
    priority[8] = 38
    priority[9] = 45
    priority[2] = 80
    projected_priority = np.max(priority[labels], axis=axis)
    projected = np.zeros(projected_priority.shape, dtype=np.uint8)
    for label, rank in enumerate(priority):
        projected[projected_priority == rank] = label
    return projected


def save_mip_qc(volume: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Save three MIP overlays."""
    colors = np.asarray(
        [
            [0, 0, 0],
            [45, 45, 45],
            [255, 255, 255],
            [255, 80, 80],
            [255, 170, 60],
            [220, 80, 220],
            [0, 0, 0],
            [80, 220, 80],
            [80, 220, 220],
            [80, 130, 255],
        ],
        dtype=np.uint8,
    )
    panels = []
    for axis in range(3):
        base = cv2.cvtColor(normalize_u8(np.max(volume, axis=axis).T), cv2.COLOR_GRAY2BGR)
        tissue = label_projection(labels, axis).T
        alpha = np.where(tissue == 1, 0.12, 0.42).astype(np.float32)[..., None]
        panel = (base.astype(np.float32) * (1.0 - alpha) + colors[tissue].astype(np.float32) * alpha).astype(np.uint8)
        cv2.putText(panel, f"MIP axis={axis}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (390, 390), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.hstack(panels))


def save_slice_qc(volume: np.ndarray, labels: np.ndarray, body: np.ndarray, path: Path) -> None:
    """Save center slice overlays through the body bbox."""
    points = np.argwhere(body)
    center = ((points.min(axis=0) + points.max(axis=0)) // 2).astype(int)
    colors = np.asarray(
        [
            [0, 0, 0],
            [45, 45, 45],
            [255, 255, 255],
            [255, 80, 80],
            [255, 170, 60],
            [220, 80, 220],
            [0, 0, 0],
            [80, 220, 80],
            [80, 220, 220],
            [80, 130, 255],
        ],
        dtype=np.uint8,
    )
    slices = [
        (volume[:, :, center[2]], labels[:, :, center[2]]),
        (volume[:, center[1], :], labels[:, center[1], :]),
        (volume[center[0], :, :], labels[center[0], :, :]),
    ]
    panels = []
    for image, tissue in slices:
        base = cv2.cvtColor(normalize_u8(image.T), cv2.COLOR_GRAY2BGR)
        tissue_t = tissue.T
        alpha = np.where(tissue_t == 1, 0.12, 0.42).astype(np.float32)[..., None]
        panel = (base.astype(np.float32) * (1.0 - alpha) + colors[tissue_t].astype(np.float32) * alpha).astype(np.uint8)
        panels.append(cv2.resize(panel, (390, 390), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(path), np.hstack(panels))


def crop_to_body(
    volume: np.ndarray,
    labels: np.ndarray,
    body: np.ndarray,
    spacing: float,
    margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Crop arrays to the visible mouse body bbox."""
    points = np.argwhere(body)
    margin = int(np.ceil(margin_mm / spacing))
    start = np.maximum(points.min(axis=0) - margin, 0)
    stop = np.minimum(points.max(axis=0) + margin + 1, volume.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))
    return volume[slices], labels[slices], [[int(lo), int(hi)] for lo, hi in zip(start, stop)]


def main() -> int:
    """Run conservative CT-visible segmentation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(args.volume))
    volume = image.get_fdata(dtype="float32")
    spacing = float(image.header.get_zooms()[0])
    smooth = ndimage.gaussian_filter(np.nan_to_num(volume), sigma=0.9)
    finite = smooth[np.isfinite(smooth)]
    body_threshold = float(np.percentile(finite, args.body_percentile))
    bone_threshold = float(np.percentile(finite, args.bone_percentile))

    bone_raw = smooth >= bone_threshold
    bone_raw = ndimage.binary_opening(bone_raw, iterations=1)
    skeleton_seed = ndimage.binary_dilation(bone_raw, iterations=args.skeleton_dilate)
    body_candidate = smooth >= body_threshold
    body = components_touching_seed(body_candidate, skeleton_seed, args.min_component_voxels)
    body = ndimage.binary_closing(body, iterations=args.body_close)
    body = ndimage.binary_fill_holes(body)
    body = largest_component(body | (skeleton_seed & body_candidate))

    points = np.argwhere(body)
    if points.size == 0:
        raise RuntimeError("Body mask is empty; lower --body-percentile or --min-component-voxels")
    bbox_min = points.min(axis=0)
    bbox_size = points.max(axis=0) - points.min(axis=0) + 1
    axis = int(np.argmax(bbox_size)) if args.head_axis == "auto" else {"x": 0, "y": 1, "z": 2}[args.head_axis]
    long_coord = np.indices(volume.shape, sparse=True)[axis]
    long_norm = (long_coord - bbox_min[axis]) / max(float(bbox_size[axis] - 1), 1.0)
    if args.head_side == "high":
        head_region = long_norm >= 0.78
        thorax_region = (long_norm >= 0.50) & (long_norm < 0.76)
        abdomen_region = (long_norm >= 0.22) & (long_norm < 0.56)
    else:
        head_region = long_norm <= 0.22
        thorax_region = (long_norm > 0.24) & (long_norm <= 0.50)
        abdomen_region = (long_norm > 0.44) & (long_norm <= 0.78)

    bone = (smooth >= bone_threshold) & body
    bone = ndimage.binary_opening(bone, iterations=1)
    bone = ndimage.binary_dilation(bone, iterations=1) & body

    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[body] = 1
    soft_high = smooth >= np.percentile(smooth[body], 58.0)
    if not args.simulate_major_organs:
        labels[body & head_region & soft_high] = 3
        labels[body & thorax_region & soft_high & ~bone] = 4
        labels[body & abdomen_region & soft_high & ~bone] = 7

    lung_threshold = float(np.percentile(smooth[body & thorax_region], args.lung_percentile))
    body_distance = ndimage.distance_transform_edt(body)
    lung = body & thorax_region & (smooth <= lung_threshold) & (body_distance >= 6.0)
    lung = ndimage.binary_opening(lung, iterations=2)
    lung_seed = ndimage.binary_dilation(bone_raw & thorax_region, iterations=5) & body
    lung = components_touching_seed(lung, lung_seed, 120)
    lung = largest_n_components(lung, n_components=4, min_size=1500)
    lung = ndimage.binary_closing(lung, iterations=1)
    labels[lung] = 9

    if args.simulate_major_organs:
        organ_allowed = body & ~bone & ~lung
        simulated_organs = {
            # Digimouse-like relative positions: heart in anterior thorax,
            # liver broad and asymmetric just caudal to thorax, stomach lateral,
            # kidneys paired and smaller in the dorsal abdomen.
            4: (
                [(0.38, 0.50, 0.49, 0.032, 0.062, 0.078), (0.41, 0.53, 0.50, 0.024, 0.046, 0.060)],
                [(0.35, 0.43, 0.42, 0.022, 0.052, 0.050)],
                41,
            ),
            7: (
                [(0.49, 0.43, 0.50, 0.060, 0.165, 0.135), (0.53, 0.55, 0.48, 0.045, 0.120, 0.110)],
                [(0.47, 0.63, 0.58, 0.035, 0.060, 0.070)],
                71,
            ),
            5: (
                [(0.57, 0.64, 0.53, 0.050, 0.095, 0.080), (0.60, 0.59, 0.50, 0.030, 0.060, 0.060)],
                [(0.55, 0.55, 0.45, 0.022, 0.040, 0.040)],
                51,
            ),
            8: (
                [(0.63, 0.34, 0.53, 0.046, 0.055, 0.078), (0.63, 0.66, 0.53, 0.046, 0.055, 0.078)],
                [],
                81,
            ),
        }
        for label, (parts, subtractors, seed) in simulated_organs.items():
            mask = digimouse_like_organ_mask(volume.shape, bbox_min, bbox_size, axis, args.head_side, parts, subtractors, seed)
            mask = ndimage.binary_opening(mask & organ_allowed, iterations=1)
            labels[mask] = label

    labels[bone] = 2

    body_only = np.where(body, volume, 0.0).astype(np.float32)
    cropped_volume, cropped_labels, crop_bbox = crop_to_body(body_only, labels, body, spacing, args.crop_margin_mm)
    cropped_body = cropped_labels > 0
    affine = image.affine.copy()
    affine[:3, 3] += np.asarray([crop_bbox[0][0], crop_bbox[1][0], crop_bbox[2][0]], dtype=np.float64) * spacing

    nib.save(nib.Nifti1Image(labels.astype(np.uint8), image.affine, image.header), args.output_dir / "visible_labels.nii.gz")
    nib.save(nib.Nifti1Image(body_only, image.affine), args.output_dir / "ct_bodyonly.nii.gz")
    nib.save(nib.Nifti1Image(cropped_volume.astype(np.float32), affine), args.output_dir / "ct_cropped.nii.gz")
    nib.save(nib.Nifti1Image(cropped_labels.astype(np.uint8), affine), args.output_dir / "visible_labels_cropped.nii.gz")
    np.save(args.output_dir / "body_mask.npy", body.astype(np.uint8))
    np.save(args.output_dir / "visible_labels.npy", labels.astype(np.uint8))
    save_mip_qc(body_only, labels, args.output_dir / "visible_labels_mip_qc.png")
    save_slice_qc(body_only, labels, body, args.output_dir / "visible_labels_qc.png")
    save_mip_qc(cropped_volume, cropped_labels, args.output_dir / "cropped_visible_labels_mip_qc.png")
    save_slice_qc(cropped_volume, cropped_labels, cropped_body, args.output_dir / "cropped_visible_labels_qc.png")

    counts = {LABELS[int(label)]: int(np.count_nonzero(labels == label)) for label in np.unique(labels)}
    manifest = {
        "version": 1,
        "source_volume": str(args.volume.resolve()),
        "status": "conservative_visible_ct_segmentation",
        "label_mapping": LABELS,
        "head_axis": "xyz"[axis],
        "head_side": args.head_side,
        "body_percentile": args.body_percentile,
        "body_threshold": body_threshold,
        "bone_percentile": args.bone_percentile,
        "bone_threshold": bone_threshold,
        "lung_percentile_in_thorax": args.lung_percentile,
        "lung_threshold": lung_threshold,
        "simulate_major_organs": args.simulate_major_organs,
        "crop_bbox_voxel_xyz": crop_bbox,
        "voxel_size_mm_from_header": spacing,
        "voxel_counts": counts,
        "provenance": {
            "body": "thresholded_soft_tissue_components_touching_dilated_ct_bone",
            "bone": "high_density_threshold_inside_body",
            "lung": "low_density_components_inside_thorax_region",
            "thresholded_nonbone_organs": "disabled when simulate_major_organs is set, except lung",
            "simulated_major_organs": "digimouse-inspired irregular multi-ellipsoid organs clipped to body and excluding CT-supported bone/lung",
        },
    }
    (args.output_dir / "segmentation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output_dir / "visible_labels_cropped.nii.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
