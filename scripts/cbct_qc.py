import argparse
import csv
import math
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import yaml


def numeric_projection_files(ct_dir: Path) -> list[tuple[float, Path]]:
    items = []
    for path in ct_dir.glob("*.tif"):
        try:
            items.append((float(path.stem), path))
        except ValueError:
            continue
    return sorted(items, key=lambda x: x[0])


def write_angle_records(ct_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = numeric_projection_files(ct_dir)
    raw_angles = [a for a, _ in raw]
    clean = [(a, p) for a, p in raw if not math.isclose(a, 360.0, abs_tol=1e-6)]
    clean_angles = [a for a, _ in clean]
    clean_int = {int(round(a)): p for a, p in clean}
    target = list(range(360))
    missing = [a for a in target if a not in clean_int]

    (out_dir / "raw_angles.txt").write_text(
        "\n".join(f"{a:g}" for a in raw_angles) + "\n", encoding="utf-8"
    )
    (out_dir / "clean_angles.txt").write_text(
        "\n".join(f"{a:g}" for a in clean_angles) + "\n", encoding="utf-8"
    )
    (out_dir / "missing_angles.txt").write_text(
        "\n".join(str(a) for a in missing) + "\n", encoding="utf-8"
    )

    clean_sorted = sorted(clean_int)
    with open(out_dir / "filled_angles.txt", "w", encoding="utf-8") as f:
        f.write("angle_deg,status,left_real,right_real,weight_right\n")
        for angle in target:
            if angle in clean_int:
                f.write(f"{angle},real,{angle},{angle},0\n")
                continue
            if not clean_sorted:
                f.write(f"{angle},missing,,,,\n")
                continue
            right_idx = np.searchsorted(clean_sorted, angle, side="right")
            if right_idx <= 0:
                left = clean_sorted[-1]
                right = clean_sorted[0]
                span = right + 360 - left
                weight = (angle + 360 - left) / span
            elif right_idx >= len(clean_sorted):
                left = clean_sorted[-1]
                right = clean_sorted[0]
                span = right + 360 - left
                weight = (angle - left) / span
            else:
                left = clean_sorted[right_idx - 1]
                right = clean_sorted[right_idx]
                weight = (angle - left) / (right - left)
            f.write(f"{angle},filled,{left},{right},{weight:.6f}\n")

    return {
        "raw_count": len(raw_angles),
        "clean_count": len(clean_angles),
        "missing_count": len(missing),
        "missing_angles": missing,
    }


def resize_for_qc(img: np.ndarray, width: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def log_projection(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    arr = np.clip(arr, 1.0, 65535.0)
    return -np.log(np.clip(arr / 65535.0, 1e-6, 1.0))


def normalize_u8(img: np.ndarray) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def plot_curves(out_path: Path, rows: list[dict], keys: list[str], title: str) -> None:
    width = 1200
    panel_h = 220
    margin_l = 90
    margin_r = 20
    margin_t = 36
    margin_b = 32
    canvas = np.full((panel_h * len(keys), width, 3), 255, dtype=np.uint8)
    angles = np.array([r["angle"] for r in rows], dtype=np.float32)
    x_min, x_max = 0.0, 359.0
    for panel_idx, key in enumerate(keys):
        y0 = panel_idx * panel_h
        panel = canvas[y0 : y0 + panel_h]
        values = np.array([r[key] for r in rows], dtype=np.float32)
        v_min = float(np.nanmin(values))
        v_max = float(np.nanmax(values))
        if v_max <= v_min:
            v_max = v_min + 1.0
        cv2.putText(
            panel,
            title if panel_idx == 0 else key,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(panel, key, (10, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(
            panel,
            (margin_l, margin_t),
            (width - margin_r, panel_h - margin_b),
            (180, 180, 180),
            1,
        )
        pts = []
        for a, v in zip(angles, values):
            x = int(margin_l + (a - x_min) / (x_max - x_min) * (width - margin_l - margin_r))
            y = int((panel_h - margin_b) - (v - v_min) / (v_max - v_min) * (panel_h - margin_t - margin_b))
            pts.append((x, y))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(panel, p0, p1, (20, 80, 220), 1)
        for p in pts[:: max(1, len(pts) // 120)]:
            cv2.circle(panel, p, 2, (20, 80, 220), -1)
        cv2.putText(panel, f"{v_max:.4g}", (margin_l + 4, margin_t + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(panel, f"{v_min:.4g}", (margin_l + 4, panel_h - margin_b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.imwrite(str(out_path), canvas)


def normalization_qc(ct_dir: Path, out_dir: Path, air_zero_percentile: float = 1.0) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = [(a, p) for a, p in numeric_projection_files(ct_dir) if not math.isclose(a, 360.0, abs_tol=1e-6)]
    rows = []
    corrected = []
    for angle, path in clean:
        img = cv2.imread(str(path), -1)
        if img is None:
            continue
        log_img = log_projection(resize_for_qc(img))
        baseline = float(np.percentile(log_img, air_zero_percentile))
        corr = np.clip(log_img - baseline, 0.0, None)
        total_before = float(np.mean(np.clip(log_img, 0.0, np.percentile(log_img, 99.5))))
        total_after = float(np.mean(np.clip(corr, 0.0, np.percentile(corr, 99.5))))
        rows.append(
            {
                "angle": int(round(angle)),
                "raw_mean": float(np.mean(img)),
                "log_mean": float(np.mean(log_img)),
                "air_baseline": baseline,
                "total_before": total_before,
                "total_after_baseline": total_after,
            }
        )
        corrected.append(corr)

    metrics = np.array([r["total_after_baseline"] for r in rows], dtype=np.float32)
    target = float(np.median(metrics[metrics > 1e-8])) if np.any(metrics > 1e-8) else 1.0
    for r in rows:
        metric = r["total_after_baseline"]
        scale = 1.0 if metric <= 1e-8 else float(np.clip(target / metric, 0.75, 1.33))
        r["mild_scale"] = scale
        r["total_after_mild_match"] = metric * scale

    with open(out_dir / "normalization_qc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    if rows:
        plot_curves(
            out_dir / "normalization_qc.png",
            rows,
            ["raw_mean", "air_baseline", "total_before", "total_after_baseline", "mild_scale"],
            "Normalization QC",
        )

    report = [
        "# Normalization QC",
        "",
        "- Single-image normalization is not applied.",
        f"- Log-domain air baseline correction percentile: {air_zero_percentile}.",
        "- Total attenuation matching is mild only; scale is clipped to [0.75, 1.33].",
        f"- Clean projection count: {len(rows)}.",
    ]
    if rows:
        scales = np.array([r["mild_scale"] for r in rows])
        report += [
            f"- Mild scale range: {scales.min():.4f} .. {scales.max():.4f}.",
            f"- Mild scale median: {np.median(scales):.4f}.",
        ]
    (out_dir / "normalization_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def motion_qc(ct_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = [(a, p) for a, p in numeric_projection_files(ct_dir) if not math.isclose(a, 360.0, abs_tol=1e-6)]
    prev = None
    rows = []
    thumbnails = []
    for angle, path in clean:
        img = cv2.imread(str(path), -1)
        if img is None:
            continue
        log_img = log_projection(resize_for_qc(img, width=256))
        u8 = normalize_u8(log_img)
        _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[idx, cv2.CC_STAT_AREA])
            x, y, w, h = stats[idx, :4]
            cx, cy = centroids[idx]
        else:
            area, x, y, w, h, cx, cy = 0, 0, 0, 0, 0, np.nan, np.nan
        if prev is None:
            diff = 0.0
            shift_x = 0.0
            shift_y = 0.0
        else:
            diff = float(np.mean(np.abs(u8.astype(np.float32) - prev.astype(np.float32))))
            shift, _ = cv2.phaseCorrelate(prev.astype(np.float32), u8.astype(np.float32))
            shift_x, shift_y = float(shift[0]), float(shift[1])
        rows.append(
            {
                "angle": int(round(angle)),
                "area": area,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w": int(w),
                "bbox_h": int(h),
                "adjacent_diff": diff,
                "phase_shift_x": shift_x,
                "phase_shift_y": shift_y,
                "phase_shift_norm": float(np.hypot(shift_x, shift_y)),
            }
        )
        thumbnails.append((int(round(angle)), u8))
        prev = u8

    if rows:
        arr_area = np.array([r["area"] for r in rows], dtype=np.float32)
        arr_diff = np.array([r["adjacent_diff"] for r in rows], dtype=np.float32)
        arr_shift = np.array([r["phase_shift_norm"] for r in rows], dtype=np.float32)
        med_area = float(np.median(arr_area))
        mad_area = float(np.median(np.abs(arr_area - med_area)) + 1e-6)
        mad_diff = float(np.median(np.abs(arr_diff - np.median(arr_diff))) + 1e-6)
        mad_shift = float(np.median(np.abs(arr_shift - np.median(arr_shift))) + 1e-6)
        for r in rows:
            area_frac = abs(r["area"] - med_area) / max(med_area, 1.0)
            area_z = abs(r["area"] - med_area) / (1.4826 * mad_area)
            diff_z = abs(r["adjacent_diff"] - float(np.median(arr_diff))) / (1.4826 * mad_diff)
            shift_z = abs(r["phase_shift_norm"] - float(np.median(arr_shift))) / (1.4826 * mad_shift)
            severe = area_z > 5.0 or shift_z > 5.0 or diff_z > 5.0
            suspicious = severe or area_z > 3.5 or shift_z > 3.5 or diff_z > 3.5
            r["area_frac_from_median"] = area_frac
            r["area_robust_z"] = area_z
            r["diff_robust_z"] = diff_z
            r["phase_shift_robust_z"] = shift_z
            r["suspicious"] = int(suspicious)
            r["severe"] = int(severe)

    with open(out_dir / "suspicious_motion_frames.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["angle"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if r.get("suspicious", 0):
                writer.writerow(r)
    with open(out_dir / "motion_qc_all_frames.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["angle"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    suspicious_angles = {r["angle"] for r in rows if r.get("suspicious", 0)}
    selected = [(a, img) for a, img in thumbnails if a in suspicious_angles]
    if not selected:
        selected = thumbnails[:: max(1, len(thumbnails) // 24)][:24]
    make_contact_sheet(selected, out_dir / "motion_contact_sheet.png")
    return {
        "suspicious_count": sum(int(r.get("suspicious", 0)) for r in rows),
        "severe_count": sum(int(r.get("severe", 0)) for r in rows),
    }


def make_contact_sheet(items: list[tuple[int, np.ndarray]], out_path: Path, cols: int = 6) -> None:
    if not items:
        return
    tile_h, tile_w = 160, 160
    rows = int(math.ceil(len(items) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, (angle, img) in enumerate(items):
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, str(angle), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = (i // cols) * tile_h
        x = (i % cols) * tile_w
        canvas[y : y + tile_h, x : x + tile_w] = tile
    cv2.imwrite(str(out_path), canvas)


def volume_qc(volume_path: Path, out_dir: Path, voxel_size: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(volume_path))
    vol = img.get_fdata(dtype="float32")
    x, y, z = [s // 2 for s in vol.shape]
    slices = [vol[:, :, z], vol[:, y, :], vol[x, :, :]]
    names = ["axial", "coronal", "sagittal"]
    panels = []
    for name, slc in zip(names, slices):
        u8 = normalize_u8(slc.T)
        u8 = cv2.resize(u8, (360, 360), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        panels.append(bgr)
    cv2.imwrite(str(out_dir / "fov_center_slices_qc.png"), np.concatenate(panels, axis=1))

    th = float(np.percentile(vol, 99.0))
    mask = vol > th
    border = (
        mask[0].sum()
        + mask[-1].sum()
        + mask[:, 0, :].sum()
        + mask[:, -1, :].sum()
        + mask[:, :, 0].sum()
        + mask[:, :, -1].sum()
    )
    border_fraction = float(border / max(mask.sum(), 1))
    fov_mm = [float(s * voxel_size) for s in vol.shape]
    report = [
        "# FOV QC",
        "",
        f"- Volume: `{volume_path.name}`",
        f"- Shape: {tuple(vol.shape)}",
        f"- Voxel size: {voxel_size} mm",
        f"- FOV: {fov_mm} mm",
        f"- High-intensity border fraction (99th percentile mask): {border_fraction:.6f}",
        "- Inspect `fov_center_slices_qc.png` for head/body clipping.",
    ]
    (out_dir / "fov_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"border_fraction": border_fraction, "shape": tuple(vol.shape), "fov_mm": fov_mm}


def denoise_and_mask(volume_path: Path, out_dir: Path) -> None:
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    from skimage import filters, measure

    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(volume_path))
    vol = img.get_fdata(dtype="float32")
    den = np.empty_like(vol, dtype=np.float32)
    for i in range(vol.shape[2]):
        den[:, :, i] = cv2.medianBlur(vol[:, :, i], 3)
    den_path = out_dir / "light_denoised_median3.nii.gz"
    nib.save(nib.Nifti1Image(den.astype(np.float32), img.affine, img.header), str(den_path))

    vals = den[den > np.percentile(den, 50)]
    threshold = float(filters.threshold_otsu(vals)) if vals.size else float(np.percentile(den, 95))
    mask = den > threshold
    labeled, n_labels = label(mask)
    if n_labels > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep = int(np.argmax(counts))
        mask = labeled == keep
    mask = binary_closing(mask, iterations=1)
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])
    np.save(out_dir / "body_mask.npy", mask.astype(np.uint8))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header), str(out_dir / "body_mask.nii.gz"))

    overlay_mask_qc(vol, mask, out_dir / "body_mask_overlay_qc.png")

    try:
        mesh_mask = mask[::2, ::2, ::2]
        verts, faces, _, _ = measure.marching_cubes(mesh_mask.astype(np.float32), level=0.5)
        verts *= 2.0
        write_ply(out_dir / "body_mask.ply", verts, faces)
    except Exception as exc:
        (out_dir / "body_mask_mesh_error.txt").write_text(str(exc), encoding="utf-8")


def overlay_mask_qc(vol: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    x, y, z = [s // 2 for s in vol.shape]
    slices = [(vol[:, :, z], mask[:, :, z]), (vol[:, y, :], mask[:, y, :]), (vol[x, :, :], mask[x, :, :])]
    panels = []
    for img, m in slices:
        u8 = normalize_u8(img.T)
        bgr = cv2.cvtColor(cv2.resize(u8, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        mu8 = cv2.resize(m.T.astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
        bgr[mu8 > 0, 1] = np.maximum(bgr[mu8 > 0, 1], 180)
        panels.append(bgr)
    cv2.imwrite(str(out_path), np.concatenate(panels, axis=1))


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load_voxel_size(config_path: Path | None, fallback: float) -> float:
    if not config_path:
        return fallback
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return float(cfg.get("ReconParam", {}).get("voxelPixelSize", fallback))


def sample_dirs(root: Path, samples: list[str]) -> list[Path]:
    if samples:
        return [root / s for s in samples]
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/foods/pro/true_data/ZJ"))
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--volume-name", default="rec_pyct_final_fov480_ramlak.nii.gz")
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--skip-mask", action="store_true")
    args = parser.parse_args()

    root_qc = args.root / "cbct_qc"
    root_qc.mkdir(exist_ok=True)
    summary = []
    voxel_size = args.voxel_size

    for sample_dir in sample_dirs(args.root, args.samples):
        ct_dir = sample_dir / "ct"
        if not ct_dir.is_dir():
            continue
        out_dir = sample_dir / "qc"
        angle_info = write_angle_records(ct_dir, out_dir)
        normalization_qc(ct_dir, out_dir)
        motion_info = motion_qc(ct_dir, out_dir)
        volume_path = sample_dir / args.volume_name
        fov_info = {}
        if volume_path.exists():
            fov_info = volume_qc(volume_path, out_dir, voxel_size)
            if not args.skip_mask:
                denoise_and_mask(volume_path, out_dir)
        clean_count = max(angle_info["clean_count"], 1)
        severe_ratio = motion_info["severe_count"] / clean_count
        suspicious_ratio = motion_info["suspicious_count"] / clean_count
        quality = "candidate_for_reconstruction"
        if angle_info["clean_count"] == 0:
            quality = "missing_ct"
        elif angle_info["missing_count"] > 90 or severe_ratio > 0.75:
            quality = "low-confidence anatomical support"
        elif severe_ratio > 0.55 or suspicious_ratio > 0.80:
            quality = "review"
        summary.append(
            {
                "sample": sample_dir.name,
                **angle_info,
                **motion_info,
                "suspicious_ratio": suspicious_ratio,
                "severe_ratio": severe_ratio,
                "quality": quality,
                "volume_exists": int(volume_path.exists()),
                "border_fraction": fov_info.get("border_fraction", np.nan),
            }
        )

    with open(root_qc / "summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary[0].keys()) if summary else ["sample"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# CBCT Summary Report", ""]
    for row in summary:
        lines += [
            f"## Sample {row['sample']}",
            f"- Clean projections: {row['clean_count']}",
            f"- Missing angles: {row['missing_count']}",
            f"- Suspicious motion frames: {row['suspicious_count']}",
            f"- Severe motion frames: {row['severe_count']}",
            f"- Center offset sanity check: manual geometry retained; only +/-2-3 px review recommended if needed.",
            f"- Quality label: {row['quality']}",
            "",
        ]
    (root_qc / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(root_qc / "summary_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
