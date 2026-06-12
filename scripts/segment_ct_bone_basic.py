#!/usr/bin/env python3
"""Basic CT bone candidate segmentation for mouse CBCT volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bone-percentile", type=float, default=99.25)
    parser.add_argument("--smooth-sigma", type=float, default=0.7)
    parser.add_argument("--min-component-voxels", type=int, default=40)
    parser.add_argument("--remove-border-components", action="store_true")
    parser.add_argument("--dilate-iterations", type=int, default=1)
    return parser.parse_args()


def normalize_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def filter_components(mask: np.ndarray, min_voxels: int, remove_border: bool) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    if remove_border:
        border_ids = set()
        for axis in range(3):
            first = [slice(None), slice(None), slice(None)]
            last = [slice(None), slice(None), slice(None)]
            first[axis] = 0
            last[axis] = mask.shape[axis] - 1
            border_ids.update(np.unique(labels[tuple(first)]).tolist())
            border_ids.update(np.unique(labels[tuple(last)]).tolist())
        for label in border_ids:
            if label != 0 and sizes[label] > min_voxels * 8:
                keep[int(label)] = False
    return keep[labels]


def label_projection(labels: np.ndarray, axis: int) -> np.ndarray:
    return np.max(labels, axis=axis).astype(np.uint8)


def save_qc(volume: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    colors = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    points = np.argwhere(labels > 0)
    if points.size:
        center = ((points.min(axis=0) + points.max(axis=0)) // 2).astype(int)
    else:
        center = np.asarray(volume.shape) // 2

    slice_defs = [
        ("axis0", volume[center[0], :, :], labels[center[0], :, :]),
        ("axis1", volume[:, center[1], :], labels[:, center[1], :]),
        ("axis2", volume[:, :, center[2]], labels[:, :, center[2]]),
    ]
    panels = []
    for name, image, tissue in slice_defs:
        base = cv2.cvtColor(normalize_u8(image.T), cv2.COLOR_GRAY2BGR)
        overlay = colors[tissue.T]
        panel = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
        cv2.putText(panel, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(output_dir / "bone_center_slices_qc.png"), np.hstack(panels))

    panels = []
    for axis in range(3):
        base = cv2.cvtColor(normalize_u8(np.max(volume, axis=axis).T), cv2.COLOR_GRAY2BGR)
        tissue = label_projection(labels, axis).T
        overlay = colors[tissue]
        panel = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
        cv2.putText(panel, f"MIP axis={axis}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
        panels.append(cv2.resize(panel, (360, 360), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(output_dir / "bone_mip_qc.png"), np.hstack(panels))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(args.volume))
    volume = image.get_fdata(dtype="float32")
    finite = volume[np.isfinite(volume)]
    nonzero = finite[finite > np.percentile(finite, 20.0)]
    values = nonzero if nonzero.size else finite
    smooth = ndimage.gaussian_filter(np.nan_to_num(volume), sigma=args.smooth_sigma)
    threshold = float(np.percentile(values, args.bone_percentile))
    bone = smooth >= threshold
    bone = ndimage.binary_opening(bone, iterations=1)
    bone = filter_components(bone, args.min_component_voxels, args.remove_border_components)
    if args.dilate_iterations > 0:
        bone = ndimage.binary_dilation(bone, iterations=args.dilate_iterations)
    labels = bone.astype(np.uint8)
    nib.save(nib.Nifti1Image(labels, image.affine, image.header), args.output_dir / "bone_labels.nii.gz")
    np.save(args.output_dir / "bone_mask.npy", labels)
    save_qc(volume, labels, args.output_dir)
    points = np.argwhere(labels > 0)
    manifest = {
        "source_volume": str(args.volume.resolve()),
        "label_mapping": {"0": "background", "1": "bone_candidate"},
        "bone_percentile": args.bone_percentile,
        "bone_threshold": threshold,
        "smooth_sigma": args.smooth_sigma,
        "min_component_voxels": args.min_component_voxels,
        "remove_border_components": args.remove_border_components,
        "dilate_iterations": args.dilate_iterations,
        "bone_voxels": int(np.count_nonzero(labels)),
        "bone_bbox_xyz": [[int(v) for v in points.min(axis=0)], [int(v) for v in points.max(axis=0)]] if points.size else None,
    }
    (args.output_dir / "bone_segmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir / "bone_labels.nii.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
