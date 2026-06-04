import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.astra.conebeam import ConeBeam


def save_preview(rec: np.ndarray, output_dir: Path) -> None:
    import cv2

    for axis in range(3):
        mip = np.max(rec, axis=axis)
        lo, hi = np.percentile(mip, [1.0, 99.8])
        if hi <= lo:
            hi = lo + 1.0
        image = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
        output = output_dir / f"mip_axis{axis}.png"
        cv2.imwrite(str(output), np.round(image * 255.0).astype(np.uint8))


def save_window_preview(rec: np.ndarray, output_dir: Path) -> None:
    """Save center-slice previews with several contrast windows."""
    import cv2

    center = [size // 2 for size in rec.shape]
    slices = [
        rec[:, :, center[2]],
        rec[:, center[1], :],
        rec[center[0], :, :],
    ]
    windows = [(0.5, 99.5), (1.0, 99.0), (5.0, 98.5), (20.0, 99.8)]
    rows = []
    for lo_q, hi_q in windows:
        panels = []
        for image in slices:
            finite = image[np.isfinite(image)]
            lo, hi = np.percentile(finite, [lo_q, hi_q])
            if hi <= lo:
                hi = lo + 1.0
            panel = np.clip((image.T - lo) / (hi - lo), 0.0, 1.0)
            panel_u8 = np.round(panel * 255.0).astype(np.uint8)
            panel_u8 = cv2.resize(panel_u8, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(
                panel_u8,
                f"p{lo_q:g}-p{hi_q:g}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                255,
                1,
                cv2.LINE_AA,
            )
            panels.append(panel_u8)
        rows.append(np.hstack(panels))
    cv2.imwrite(str(output_dir / "window_center_slices_qc.png"), np.vstack(rows))


def save_fixed_window_preview(
    rec: np.ndarray,
    output_dir: Path,
    windows: dict[str, tuple[float, float]],
    prefix: str = "fixed",
) -> None:
    import cv2

    center = [size // 2 for size in rec.shape]
    slices = [
        rec[:, :, center[2]],
        rec[:, center[1], :],
        rec[center[0], :, :],
    ]
    rows = []
    for name, (lo, hi) in windows.items():
        if hi <= lo:
            hi = lo + 1.0
        panels = []
        for image in slices:
            panel = np.clip((image.T - lo) / (hi - lo), 0.0, 1.0)
            panel_u8 = np.round(panel * 255.0).astype(np.uint8)
            panel_u8 = cv2.resize(panel_u8, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(
                panel_u8,
                f"{name} {lo:g}-{hi:g}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                255,
                1,
                cv2.LINE_AA,
            )
            panels.append(panel_u8)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(output_dir / f"{prefix}_window_center_slices_qc.png"), np.vstack(rows))


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def body_roi_mask(rec: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    threshold = float(np.percentile(rec, 67.0))
    mask = largest_component(rec > threshold)
    if not np.any(mask):
        mask = rec > threshold
    points = np.argwhere(mask)
    if points.size == 0:
        bounds = [[0, size] for size in rec.shape]
        return np.ones(rec.shape, dtype=bool), bounds
    lo = np.maximum(points.min(axis=0) - 8, 0)
    hi = np.minimum(points.max(axis=0) + 9, rec.shape)
    bounds = [[int(a), int(b)] for a, b in zip(lo, hi)]
    roi = np.zeros(rec.shape, dtype=bool)
    roi[tuple(slice(a, b) for a, b in bounds)] = True
    return roi, bounds


def save_roi_rolling_qc(rec: np.ndarray, output_dir: Path) -> tuple[list[list[int]], dict]:
    import cv2

    roi, bounds = body_roi_mask(rec)
    roi_values = rec[roi]
    windows = {
        "wide": (float(np.percentile(roi_values, 0.5)), float(np.percentile(roi_values, 99.7))),
        "soft": (float(np.percentile(roi_values, 35.0)), float(np.percentile(roi_values, 97.5))),
        "bone": (float(np.percentile(roi_values, 90.0)), float(np.percentile(roi_values, 99.85))),
    }
    for name, (lo, hi) in windows.items():
        if hi <= lo:
            windows[name] = (lo, lo + 1.0)

    for window_name, (lo, hi) in windows.items():
        for axis in range(3):
            start, stop = bounds[axis]
            indices = np.linspace(start, max(start, stop - 1), 9).round().astype(int)
            panels = []
            for idx in indices:
                if axis == 0:
                    image = rec[idx, :, :]
                elif axis == 1:
                    image = rec[:, idx, :]
                else:
                    image = rec[:, :, idx]
                panel = np.clip((image.T - lo) / (hi - lo), 0.0, 1.0)
                panel_u8 = np.round(panel * 255.0).astype(np.uint8)
                panel_u8 = cv2.resize(panel_u8, (180, 180), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    panel_u8,
                    f"{axis}:{idx}",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    255,
                    1,
                    cv2.LINE_AA,
                )
                panels.append(panel_u8)
            cv2.imwrite(str(output_dir / f"roi_{window_name}_axis{axis}_slices.png"), np.hstack(panels))
    return bounds, {key: {"low": value[0], "high": value[1]} for key, value in windows.items()}


def volume_quality_metrics(rec: np.ndarray) -> dict:
    from scipy import ndimage

    roi, bounds = body_roi_mask(rec)
    roi_values = rec[roi]
    if roi_values.size == 0:
        roi_values = rec[np.isfinite(rec)]
    bone_threshold = float(np.percentile(roi_values, 98.5))
    soft_low, soft_high = np.percentile(roi_values, [35.0, 85.0])
    bone = (rec > bone_threshold) & roi
    soft = (rec >= soft_low) & (rec <= soft_high) & roi
    background = ~ndimage.binary_dilation(roi, iterations=5)
    gx, gy, gz = np.gradient(rec.astype(np.float32, copy=False))
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    edge_band = ndimage.binary_dilation(bone, iterations=2) & ~ndimage.binary_erosion(bone, iterations=1)
    return {
        "roi_bounds": bounds,
        "roi_voxels": int(np.count_nonzero(roi)),
        "bone_threshold_p98_5": bone_threshold,
        "bone_edge_gradient_p95": float(np.percentile(grad[edge_band], 95.0)) if np.any(edge_band) else 0.0,
        "soft_mean": float(np.mean(rec[soft])) if np.any(soft) else 0.0,
        "soft_std": float(np.std(rec[soft])) if np.any(soft) else 0.0,
        "soft_local_contrast_p90_p10": float(np.percentile(roi_values, 90.0) - np.percentile(roi_values, 10.0)),
        "background_std": float(np.std(rec[background])) if np.any(background) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--size-x", type=int)
    parser.add_argument("--size-y", type=int)
    parser.add_argument("--size-z", type=int)
    parser.add_argument("--voxel-size", type=float, default=0.3)
    parser.add_argument("--roi-x", type=float, default=0.0)
    parser.add_argument("--roi-y", type=float, default=0.0)
    parser.add_argument("--roi-z", type=float, default=0.0)
    parser.add_argument("--ring-kernel-size", type=int, default=9)
    parser.add_argument("--filter", default="Hamming", choices=["Ram-Lak", "Shepp-Logan", "Cosine", "Hamming", "Hann"])
    parser.add_argument("--fill-missing-degrees", action="store_true")
    parser.add_argument("--air-percentile", type=float)
    parser.add_argument("--air-zero-percentile", type=float)
    parser.add_argument("--mass-normalize", action="store_true")
    parser.add_argument("--projection-gaussian-sigma", type=float)
    parser.add_argument("--projection-median-kernel", type=int, default=0)
    parser.add_argument("--intensity-floor", type=float, default=1.0)
    parser.add_argument("--intensity-clip-percentile", type=float)
    parser.add_argument("--fixed-soft-window", nargs=2, type=float)
    parser.add_argument("--fixed-bone-window", nargs=2, type=float)
    parser.add_argument(
        "--detector-scale",
        type=float,
        default=None,
        help="Projection resize scale. 0.5 keeps the legacy half-resolution path; 1.0 uses raw TIFF resolution.",
    )
    args = parser.parse_args()

    calibration_path = args.calibration or args.proj_dir / "calibration_vshift.json"
    with open(calibration_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    recon = cfg["ReconParam"]
    cal_params = cfg["CalibrationParam"]
    detector_scale = float(args.detector_scale if args.detector_scale is not None else calib.get("sx", 0.5))
    detector_width = int(cal_params["detectorWidth"])
    detector_height = int(cal_params["detectorHeight"])
    detector_columns = int(round(detector_width * detector_scale))
    detector_rows = int(round(detector_height * detector_scale))
    vc_raw = float(calib.get("vc_raw", float(calib.get("vc_recon", 0.0)) / max(float(calib.get("sx", detector_scale)), 1.0e-8)))
    vs_raw = float(calib.get("vs_raw", float(calib.get("vs_recon", 0.0)) / max(float(calib.get("sy", detector_scale)), 1.0e-8)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nx = int(args.size_x if args.size_x is not None else args.size)
    ny = int(args.size_y if args.size_y is not None else args.size)
    nz = int(args.size_z if args.size_z is not None else args.size)
    cb = ConeBeam(
        SOD=float(calib["SOD"]),
        SDD=float(calib["SDD"]),
        NX=nx,
        NY=ny,
        NZ=nz,
        TM=detector_rows,
        TN=detector_columns,
        dd_row=float(recon["ySpacing"]),
        dd_column=float(recon["xSpacing"]),
        voxel_size=args.voxel_size,
        number_of_img=360,
        proj_path=str(args.proj_dir),
        detectorX=float(calib["u0_used"]),
        detectorY=float(calib["v0_used"]),
        pixel_size_raw=float(cal_params["detectorPixelSize"]),
        sx=detector_scale,
        sy=detector_scale,
        eta=float(calib["eta"]),
        vc=vc_raw * detector_scale,
        vs=vs_raw * detector_scale,
        rotation=float(calib["detector_roll_deg"]),
        angle_offset_deg=float(calib.get("angle_offset_deg", 0.0)),
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        vol_center_x=args.roi_x,
        vol_center_y=args.roi_y,
        vol_center_z=args.roi_z,
        air_percentile=args.air_percentile,
        air_zero_percentile=args.air_zero_percentile,
        mass_normalize=args.mass_normalize,
        projection_gaussian_sigma=args.projection_gaussian_sigma,
        projection_median_kernel=args.projection_median_kernel,
        intensity_floor=args.intensity_floor,
        intensity_clip_percentile=args.intensity_clip_percentile,
    )
    cb.load_img(
        angle_from_filename=True,
        drop_duplicate_360=True,
        fill_missing_degrees=args.fill_missing_degrees,
    )
    rec, use_cuda = cb.reconstruct(
        filter_type=args.filter,
        algorithm="FDK",
        ring_correction=True,
        ring_kernel_size=args.ring_kernel_size,
    )
    affine = np.diag([args.voxel_size, args.voxel_size, args.voxel_size, 1.0])
    filter_name = args.filter.lower().replace("-", "")
    nii_path = args.output_dir / f"rec_vshift_{filter_name}.nii.gz"
    nib.save(nib.Nifti1Image(rec.astype(np.float32, copy=False), affine), nii_path)
    save_preview(rec, args.output_dir)
    save_window_preview(rec, args.output_dir)
    fixed_windows = {}
    if args.fixed_soft_window is not None:
        fixed_windows["soft"] = (args.fixed_soft_window[0], args.fixed_soft_window[1])
    if args.fixed_bone_window is not None:
        fixed_windows["bone"] = (args.fixed_bone_window[0], args.fixed_bone_window[1])
    if fixed_windows:
        save_fixed_window_preview(rec, args.output_dir, fixed_windows)
    roi_bounds, roi_windows = save_roi_rolling_qc(rec, args.output_dir)
    metrics = volume_quality_metrics(rec)
    metadata = {
        "projection_dir": str(args.proj_dir.resolve()),
        "calibration": str(calibration_path.resolve()),
        "shape": list(rec.shape),
        "requested_shape": [nx, ny, nz],
        "voxel_size_mm": args.voxel_size,
        "detector_scale": detector_scale,
        "detector_shape_rows_cols": [detector_rows, detector_columns],
        "filter": args.filter,
        "ring_correction": True,
        "ring_kernel_size": args.ring_kernel_size,
        "preprocess": {
            "air_percentile": args.air_percentile,
            "air_zero_percentile": args.air_zero_percentile,
            "mass_normalize": args.mass_normalize,
            "projection_gaussian_sigma": args.projection_gaussian_sigma,
            "projection_median_kernel": args.projection_median_kernel,
            "intensity_floor": args.intensity_floor,
            "intensity_clip_percentile": args.intensity_clip_percentile,
        },
        "projection_preprocess_summary": cb.projection_preprocess_summary(),
        "roi_bounds": roi_bounds,
        "roi_windows": roi_windows,
        "fixed_windows": {
            key: {"low": value[0], "high": value[1]} for key, value in fixed_windows.items()
        },
        "quality_metrics": metrics,
        "vc_used": vc_raw * detector_scale,
        "vs_used": vs_raw * detector_scale,
    }
    (args.output_dir / "reconstruction_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Done] {nii_path} shape={rec.shape} cuda={use_cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
