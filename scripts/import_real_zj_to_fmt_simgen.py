#!/usr/bin/env python3
"""Build an FMT-SimGen-compatible bundle from real ZJ CT and fluorescence data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage


DEFAULT_ANGLES = list(range(-90, 91, 10))
MODEL_ANGLES = [-90, -60, -30, 0, 30, 60, 90]


def load_processed_projection(npz_path: Path) -> tuple[np.ndarray, int]:
    """Load processed fluorescence frames and fuse multiple frames by median."""
    data = np.load(npz_path, allow_pickle=True)
    if "processed" not in data:
        raise ValueError(f"{npz_path} does not contain a processed array")
    frames = np.asarray(data["processed"], dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3:
        raise ValueError(f"Expected processed shape (frames,H,W), got {frames.shape} in {npz_path}")
    return np.median(frames, axis=0).astype(np.float32), int(frames.shape[0])


def load_real_projections(flu_npz_dir: Path, angles: list[int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load numbered fluorescence NPZ files as angle-keyed real projections."""
    files = sorted(flu_npz_dir.glob("*.npz"), key=lambda path: int(path.stem))
    if len(files) != len(angles):
        raise ValueError(f"Expected {len(angles)} fluorescence NPZ files, found {len(files)} in {flu_npz_dir}")

    projections: dict[str, np.ndarray] = {}
    sources: list[dict[str, Any]] = []
    for angle, path in zip(angles, files):
        projection, frame_count = load_processed_projection(path)
        projections[str(angle)] = projection
        sources.append(
            {
                "angle_deg": angle,
                "source": str(path),
                "source_index": int(path.stem),
                "frame_count": frame_count,
                "fusion": "pixelwise_median" if frame_count > 1 else "single_frame",
            }
        )
    return projections, {"sources": sources}


def subtract_static_reflection(
    projections: dict[str, np.ndarray],
    percentile: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Remove camera-fixed background and reflection estimated across turntable angles."""
    stack = np.stack(list(projections.values()), axis=0).astype(np.float32)
    background = np.percentile(stack, percentile, axis=0).astype(np.float32)
    corrected = {
        angle: np.clip(image - background, 0.0, None).astype(np.float32)
        for angle, image in projections.items()
    }
    return corrected, background


def crop_resize_projection(
    projection: np.ndarray,
    center_u_px: float,
    center_v_px: float,
    camera_pixel_size_mm: float,
    fov_mm: float,
    output_resolution: int,
) -> np.ndarray:
    """Crop a physical square FOV from the real camera and resize it for model input."""
    crop_size_px = max(1, int(round(fov_mm / camera_pixel_size_mm)))
    x0 = int(round(center_u_px - crop_size_px / 2.0))
    y0 = int(round(center_v_px - crop_size_px / 2.0))
    x1 = x0 + crop_size_px
    y1 = y0 + crop_size_px
    crop = np.zeros((crop_size_px, crop_size_px), dtype=np.float32)
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, projection.shape[1])
    src_y1 = min(y1, projection.shape[0])
    if src_x0 < src_x1 and src_y0 < src_y1:
        crop[src_y0 - y0 : src_y1 - y0, src_x0 - x0 : src_x1 - x0] = projection[
            src_y0:src_y1, src_x0:src_x1
        ]
    return cv2.resize(
        crop,
        (output_resolution, output_resolution),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def build_model_projections(
    corrected: dict[str, np.ndarray],
    angles: list[int],
    center_u_px: float,
    center_v_px: float,
    camera_pixel_size_mm: float,
    fov_mm: float,
    output_resolution: int,
) -> dict[str, np.ndarray]:
    """Select the seven MCX-compatible views and map them into the model detector frame."""
    projections: dict[str, np.ndarray] = {}
    for angle in angles:
        key = str(angle)
        if key not in corrected:
            raise KeyError(f"Corrected real fluorescence is missing model angle {key}")
        projections[key] = crop_resize_projection(
            corrected[key],
            center_u_px=center_u_px,
            center_v_px=center_v_px,
            camera_pixel_size_mm=camera_pixel_size_mm,
            fov_mm=fov_mm,
            output_resolution=output_resolution,
        )
    return projections


def normalize_png(image: np.ndarray) -> np.ndarray:
    """Convert a projection to uint8 for visual QA without changing saved arrays."""
    lo, hi = np.percentile(image, [1.0, 99.7])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)


def save_projection_montage(
    path: Path,
    projections: dict[str, np.ndarray],
    angles: list[int],
    tile_size: tuple[int, int] = (256, 205),
) -> None:
    """Save an angle-labelled projection montage for manual review."""
    tiles: list[np.ndarray] = []
    for angle in angles:
        image = normalize_png(projections[str(angle)])
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            f"{angle:+d}",
            (6, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            255,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    columns = 5
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    cv2.imwrite(str(path), np.vstack(rows))


def load_mask(mask_path: Path | None, ct: np.ndarray, threshold: float | None) -> tuple[np.ndarray, str]:
    """Load an existing body mask or generate a provisional largest-component mask."""
    if mask_path is not None:
        mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0
        if mask.shape != ct.shape:
            raise ValueError(f"Body mask shape {mask.shape} does not match CT shape {ct.shape}")
        return mask, "provided_body_mask"

    finite = ct[np.isfinite(ct)]
    if threshold is None:
        # The lower tail contains weak reconstruction haze connected to the rack.
        # A high threshold separates the cylindrical phantom from its thin support.
        threshold = float(np.percentile(finite, 95.0))
    candidate = np.isfinite(ct) & (ct > threshold)
    labels, count = ndimage.label(candidate)
    if count == 0:
        raise ValueError("Threshold body segmentation is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=2.0) >= 0.5
    mask = ndimage.binary_fill_holes(mask)
    return mask, f"provisional_smoothed_largest_component_threshold:{threshold:.8g}"


def orient_ct_to_fmt(volume: np.ndarray, input_array_order: str) -> np.ndarray:
    """Convert stored CT voxels to FMT XYZ using the provisional hardware layout prior."""
    if input_array_order == "astra_zyx":
        ct_xyz = np.transpose(volume, (2, 1, 0))
    elif input_array_order == "xyz":
        ct_xyz = volume
    else:
        raise ValueError(f"Unsupported CT array order: {input_array_order}")
    return np.flip(np.transpose(ct_xyz, (0, 2, 1)), axis=2)


def crop_to_mask(
    ct: np.ndarray,
    body_mask: np.ndarray,
    voxel_size_mm: float,
    margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Crop CT and mask to the body bounding box plus an isotropic margin."""
    points = np.argwhere(body_mask)
    if len(points) == 0:
        raise ValueError("Body mask is empty")
    margin = int(np.ceil(margin_mm / voxel_size_mm))
    start = np.maximum(points.min(axis=0) - margin, 0)
    stop = np.minimum(points.max(axis=0) + margin + 1, body_mask.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))
    return ct[slices], body_mask[slices], slices


def affine_for_voxel_size(voxel_size_mm: float) -> np.ndarray:
    """Return a corner-origin isotropic NIfTI affine."""
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = voxel_size_mm
    affine[1, 1] = voxel_size_mm
    affine[2, 2] = voxel_size_mm
    return affine


def save_nifti(path: Path, array: np.ndarray, voxel_size_mm: float) -> None:
    """Save an XYZ array with an explicit isotropic voxel size."""
    image = nib.Nifti1Image(array, affine_for_voxel_size(voxel_size_mm))
    image.header.set_zooms((voxel_size_mm, voxel_size_mm, voxel_size_mm))
    nib.save(image, str(path))


def load_fluorescence_calibration(mask_points_csv: Path | None) -> dict[str, Any]:
    """Load the approved mouse-rack calibration derived from 20_Mask.tif."""
    calibration: dict[str, Any] = {
        "reference_mask": "20_Mask.tif",
        "reference_position": "cube_in_mouse_rack_at_turntable_center",
        "bead_spacing_mm": 10.0,
        "camera_distance_mm": 315.0,
        "bead_centers_px": [
            [374.0, 78.0],
            [374.0, 117.0],
            [376.0, 153.0],
            [374.0, 191.0],
            [374.0, 228.0],
            [374.0, 268.0],
        ],
    }
    points = np.asarray(calibration["bead_centers_px"], dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    calibration["mean_bead_spacing_px"] = round(float(distances.mean()), 6)
    calibration["detector_pixel_size_mm"] = round(float(10.0 / distances.mean()), 8)
    calibration["line_center_px"] = points.mean(axis=0).round(6).tolist()
    calibration["source_csv"] = str(mask_points_csv) if mask_points_csv is not None else None
    return calibration


def build_subject_manifest(
    subject_id: str,
    segmentation_path: Path,
    output_dir: Path,
    shape_xyz: tuple[int, int, int],
    voxel_size_mm: float,
    label_mapping: dict[int, int],
) -> dict[str, Any]:
    """Build a SubjectManifest-compatible JSON document."""
    extent = (np.asarray(shape_xyz, dtype=np.float64) * voxel_size_mm).tolist()
    return {
        "subject_id": subject_id,
        "world_frame": "mcx_trunk_local_mm",
        "output_dir": str(output_dir),
        "mcx_volume": {
            "shape_xyz": list(shape_xyz),
            "shape_zyx": [shape_xyz[2], shape_xyz[1], shape_xyz[0]],
            "voxel_size_mm": voxel_size_mm,
            "origin_world_mm": [0.0, 0.0, 0.0],
            "extent_mm": extent,
            "bbox_world_mm": {"min": [0.0, 0.0, 0.0], "max": extent},
        },
        "volume_center_world_mm": (np.asarray(extent) / 2.0).tolist(),
        "voxel_grid_gt": {
            "shape": list(shape_xyz),
            "spacing_mm": voxel_size_mm,
            "offset_world_mm": [0.0, 0.0, 0.0],
            "frame": "mcx_trunk_local_mm",
        },
        "atlas_to_world_offset_mm": [0.0, 0.0, 0.0],
        "segmentation_path": str(segmentation_path),
        "segmentation_format": "nifti",
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "label_roles": {
            "background_labels": [0],
            "allowed_tumor_labels": sorted(value for value in label_mapping.values() if value > 0),
            "forbidden_tumor_labels": [0],
        },
    }


def update_dataset_manifest(output_root: Path, subject_id: str, sample_id: str) -> None:
    """Create or extend the real-data dataset manifest without dropping subjects."""
    path = output_root / "dataset_manifest.json"
    subjects: list[str] = []
    samples: list[str] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        subjects = [str(value) for value in existing.get("subjects", [])]
        samples = [str(value) for value in existing.get("samples", [])]
    if subject_id not in subjects:
        subjects.append(subject_id)
    if sample_id not in samples:
        samples.append(sample_id)
    manifest = {
        "version": 1,
        "kind": "real_zj_ct_fluorescence",
        "subjects": sorted(subjects),
        "samples": sorted(samples),
        "samples_dir": str((output_root / "samples").resolve()),
        "ground_truth_available": False,
        "purpose": "real_fluorescence_test_and_simulation_quality_comparison",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_inference_split(output_root: Path, sample_id: str) -> None:
    """Register a real sample separately from simulated train/validation splits."""
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / "inference.txt"
    samples = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if sample_id not in samples:
        samples.append(sample_id)
    path.write_text("".join(f"{value}\n" for value in sorted(samples)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--flu-npz-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--body-mask", type=Path)
    parser.add_argument(
        "--tissue-labels",
        type=Path,
        help="Optional Digimouse-like label NIfTI. Values 1..9 are preserved for MCX.",
    )
    parser.add_argument("--ct-voxel-size-mm", type=float, default=0.25)
    parser.add_argument("--ct-array-order", choices=["astra_zyx", "xyz"], default="astra_zyx")
    parser.add_argument("--body-threshold", type=float)
    parser.add_argument("--crop-margin-mm", type=float, default=2.0)
    parser.add_argument("--angles", type=int, nargs="+", default=DEFAULT_ANGLES)
    parser.add_argument("--mask-points-csv", type=Path)
    parser.add_argument("--static-background-percentile", type=float, default=20.0)
    parser.add_argument("--flu-center-u-px", type=float, default=392.0)
    parser.add_argument("--flu-center-v-px", type=float, default=312.0)
    parser.add_argument("--flu-camera-pixel-size-mm", type=float)
    parser.add_argument("--model-fov-mm", type=float, default=80.0)
    parser.add_argument("--model-resolution", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    """Create one real-data subject bundle."""
    args = parse_args()
    sample_id = args.sample_id or f"sample_{args.subject_id}"
    if not sample_id.startswith("sample_"):
        raise ValueError("sample-id must start with 'sample_' for downstream dataset discovery")
    subject_dir = args.output_root / "subjects" / args.subject_id
    sample_dir = args.output_root / "samples" / sample_id
    shared_dir = subject_dir / "shared"
    for path in (subject_dir, sample_dir, shared_dir):
        path.mkdir(parents=True, exist_ok=True)

    ct_image = nib.load(str(args.ct))
    ct_native = np.asarray(ct_image.dataobj, dtype=np.float32)
    if args.tissue_labels is not None:
        labels_native = np.asarray(nib.load(str(args.tissue_labels)).dataobj, dtype=np.uint8)
        if labels_native.shape != ct_native.shape:
            raise ValueError(f"Tissue labels shape {labels_native.shape} does not match CT shape {ct_native.shape}")
        mask_native = labels_native > 0
        segmentation_method = "provided_digimouse_like_tissue_labels"
    else:
        mask_native, segmentation_method = load_mask(args.body_mask, ct_native, args.body_threshold)
        labels_native = mask_native.astype(np.uint8)
    ct_fmt = orient_ct_to_fmt(ct_native, args.ct_array_order)
    mask_fmt = orient_ct_to_fmt(mask_native, args.ct_array_order)
    labels_fmt = orient_ct_to_fmt(labels_native, args.ct_array_order)
    ct_crop, mask_crop, crop_slices = crop_to_mask(
        ct_fmt,
        mask_fmt,
        voxel_size_mm=args.ct_voxel_size_mm,
        margin_mm=args.crop_margin_mm,
    )
    labels_crop = labels_fmt[crop_slices]
    label_values = sorted(int(value) for value in np.unique(labels_crop))
    label_mapping = {value: value for value in label_values}

    ct_path = subject_dir / "ct_canonical.nii.gz"
    segmentation_path = shared_dir / "body_labels.nii.gz"
    atlas_path = shared_dir / "atlas_labels.npz"
    save_nifti(ct_path, ct_crop.astype(np.float32), args.ct_voxel_size_mm)
    save_nifti(segmentation_path, labels_crop.astype(np.uint8), args.ct_voxel_size_mm)
    np.savez_compressed(
        atlas_path,
        original_labels=labels_crop.astype(np.uint8),
        voxel_size=np.float32(args.ct_voxel_size_mm),
    )
    shutil.copy2(ct_path, sample_dir / "ct_canonical.nii.gz")
    shutil.copy2(segmentation_path, sample_dir / "body_labels.nii.gz")

    calibration = load_fluorescence_calibration(args.mask_points_csv)
    camera_pixel_size_mm = args.flu_camera_pixel_size_mm or calibration["detector_pixel_size_mm"]
    projections, projection_meta = load_real_projections(args.flu_npz_dir, args.angles)
    corrected, static_background = subtract_static_reflection(
        projections,
        percentile=args.static_background_percentile,
    )
    model_projections = build_model_projections(
        corrected,
        angles=MODEL_ANGLES,
        center_u_px=args.flu_center_u_px,
        center_v_px=args.flu_center_v_px,
        camera_pixel_size_mm=camera_pixel_size_mm,
        fov_mm=args.model_fov_mm,
        output_resolution=args.model_resolution,
    )
    np.savez_compressed(sample_dir / "proj.npz", **model_projections)
    np.savez_compressed(sample_dir / "proj_real_processed_full.npz", **projections)
    np.savez_compressed(sample_dir / "proj_real_corrected_full.npz", **corrected)
    np.save(sample_dir / "reflection_static_background.npy", static_background)
    save_projection_montage(sample_dir / "proj_real_before.png", projections, args.angles)
    save_projection_montage(sample_dir / "proj_real_corrected.png", corrected, args.angles)
    save_projection_montage(sample_dir / "proj_model_7view.png", model_projections, MODEL_ANGLES)
    cv2.imwrite(str(sample_dir / "reflection_static_background.png"), normalize_png(static_background))

    crop_size_px = int(round(args.model_fov_mm / camera_pixel_size_mm))
    preprocess = {
        "version": 1,
        "purpose": "real_fluorescence_test_preprocessing",
        "training_data_source": "MCX simulated projection and simulated fluorescence volume",
        "test_data_source": "real processed fluorescence camera projection",
        "full_angles_deg": args.angles,
        "model_angles_deg": MODEL_ANGLES,
        "static_reflection_correction": {
            "method": "subtract_pixelwise_angular_percentile_then_clip_nonnegative",
            "angular_percentile": args.static_background_percentile,
            "reason": "remove camera-fixed right-side reflection without deleting rotating source signal",
        },
        "camera_pixel_size_mm": camera_pixel_size_mm,
        "model_detector": {
            "fov_mm": args.model_fov_mm,
            "resolution_hw": [args.model_resolution, args.model_resolution],
            "center_uv_px_in_real_camera": [args.flu_center_u_px, args.flu_center_v_px],
            "crop_size_px_in_real_camera": crop_size_px,
            "crop_bounds_xyxy_in_real_camera": [
                int(round(args.flu_center_u_px - crop_size_px / 2.0)),
                int(round(args.flu_center_v_px - crop_size_px / 2.0)),
                int(round(args.flu_center_u_px - crop_size_px / 2.0)) + crop_size_px,
                int(round(args.flu_center_v_px - crop_size_px / 2.0)) + crop_size_px,
            ],
        },
        "spatial_status": "provisional_until_manual_review_of_model_7view_png",
    }
    (sample_dir / "projection_preprocess.json").write_text(
        json.dumps(preprocess, indent=2),
        encoding="utf-8",
    )

    registration = {
        "version": 1,
        "subject_id": args.subject_id,
        "status": "provisional_layout_prior",
        "fluorescence_calibration": calibration,
        "ct_native_to_fmt_canonical": {
            "input_array_order": args.ct_array_order,
            "astra_reconstruction_array_order": "ZYX",
            "axis_mapping": {"FMT_X": "+CT_X", "FMT_Y": "+CT_Z", "FMT_Z": "-CT_Y"},
            "reason": "camera_at_6_oclock_detector_at_9_oclock_source_at_3_oclock",
        },
        "real_projection_contract": {
            "raw_full_archive": "proj_real_processed_full.npz",
            "corrected_full_archive": "proj_real_corrected_full.npz",
            "model_input_archive": "proj.npz",
            "array_order": "image_vu",
            "raw_shape_hw": list(next(iter(projections.values())).shape),
            "model_shape_hw": list(next(iter(model_projections.values())).shape),
            "value_source": "LightField NPZ processed channel",
            "multi_frame_fusion": "pixelwise_median",
            "normalization": "none",
            "angle_status": "layout_prior_unverified",
            "preprocess": str((sample_dir / "projection_preprocess.json").resolve()),
        },
        "crop_slices_fmt_xyz": [
            [crop_slices[0].start, crop_slices[0].stop],
            [crop_slices[1].start, crop_slices[1].stop],
            [crop_slices[2].start, crop_slices[2].stop],
        ],
        "assumptions": [
            "20_Mask.tif and CT cube acquisition used the same unmoved mouse rack.",
            "The mouse is fixed relative to the rack while the rack rotates with the turntable.",
            "The six collinear beads calibrate vertical scale and offset but do not fully constrain 3D pose.",
            "Horizontal alignment and zero-angle orientation use the hardware clock-position prior.",
            "Real fluorescence source indices 1..19 map to angles -90..90 degrees in 10-degree steps unless overridden.",
        ],
    }
    (subject_dir / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")

    subject_manifest = build_subject_manifest(
        args.subject_id,
        segmentation_path.resolve(),
        shared_dir.resolve(),
        tuple(int(v) for v in ct_crop.shape),
        args.ct_voxel_size_mm,
        label_mapping,
    )
    (shared_dir / "frame_manifest.json").write_text(json.dumps(subject_manifest, indent=2), encoding="utf-8")

    sample_manifest = {
        "subject_id": args.subject_id,
        "sample_id": sample_id,
        "kind": "real_ct_real_fluorescence_inference",
        "ground_truth_available": False,
        "purpose": "test_real_projection_and_compare_against_simulated_projection",
        "ct_source": str(args.ct),
        "ct_voxel_size_mm": args.ct_voxel_size_mm,
        "ct_array_order": args.ct_array_order,
        "segmentation_method": segmentation_method,
        "fluorescence": projection_meta,
        "registration": str((subject_dir / "registration.json").resolve()),
        "subject_manifest": str((shared_dir / "frame_manifest.json").resolve()),
    }
    (sample_dir / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")

    shape_xyz = tuple(int(v) for v in ct_crop.shape)
    extent = (np.asarray(shape_xyz, dtype=np.float64) * args.ct_voxel_size_mm).tolist()
    config = {
        "_base_": "/home/foods/pro/FMT-SimGen/config/default.yaml",
        "subject": {
            "id": args.subject_id,
            "manifest_path": str((shared_dir / "frame_manifest.json").resolve()),
            "segmentation_path": str(segmentation_path.resolve()),
            "format": "nifti",
            "output_dir": str(shared_dir.resolve()),
            "target_voxel_size_mm": args.ct_voxel_size_mm,
            "volume_shape_xyz": list(shape_xyz),
            "crop_bbox_mm": {
                "x": [0.0, extent[0]],
                "y": [0.0, extent[1]],
                "z": [0.0, extent[2]],
            },
            "label_mapping": label_mapping,
            "label_roles": {
                "background_labels": [0],
                "allowed_tumor_labels": sorted(value for value in label_mapping.values() if value > 0),
                "forbidden_tumor_labels": [0],
            },
        },
        "atlas": {
            "path": str(atlas_path.resolve()),
            "voxel_size": args.ct_voxel_size_mm,
            "tissue_merge": label_mapping,
        },
        "mesh": {
            "output_path": str(shared_dir.resolve()),
            "mesh_file": str((shared_dir / f"{args.subject_id}_mesh.npz").resolve()),
        },
        "mcx": {
            "volume_path": str((shared_dir / "mcx_volume_trunk.bin").resolve()),
            "material_path": str((shared_dir / "mcx_material.yaml").resolve()),
            "trunk_crop": {"y_start": 0, "y_end": shape_xyz[1]},
            "downsample_factor": 1,
            "voxel_size_mm": args.ct_voxel_size_mm,
            "volume_shape": [shape_xyz[2], shape_xyz[1], shape_xyz[0]],
            "num_tissues": max(label_mapping.values()) + 1,
            "tissue_mapping": label_mapping,
        },
        "dataset": {
            "experiment_name": f"real_{args.subject_id}",
            "output_path": str(args.output_root.resolve()),
        },
    }
    (subject_dir / "fmt_simgen_subject.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    update_dataset_manifest(args.output_root, args.subject_id, sample_id)
    update_inference_split(args.output_root, sample_id)

    print(f"Wrote real subject bundle: {subject_dir}")
    print(f"Wrote sample: {sample_dir}")
    print(f"Canonical CT shape: {ct_crop.shape}, voxel={args.ct_voxel_size_mm} mm")
    print(f"Real fluorescence projections: full={len(projections)}, model={len(model_projections)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
