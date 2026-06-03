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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proj_dir", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--voxel-size", type=float, default=0.3)
    parser.add_argument("--roi-x", type=float, default=0.0)
    parser.add_argument("--roi-y", type=float, default=0.0)
    parser.add_argument("--roi-z", type=float, default=0.0)
    parser.add_argument("--ring-kernel-size", type=int, default=9)
    parser.add_argument("--filter", default="Hamming", choices=["Ram-Lak", "Shepp-Logan", "Cosine", "Hamming", "Hann"])
    parser.add_argument("--fill-missing-degrees", action="store_true")
    args = parser.parse_args()

    calibration_path = args.calibration or args.proj_dir / "calibration_vshift.json"
    with open(calibration_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    recon = cfg["ReconParam"]
    cal_params = cfg["CalibrationParam"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cb = ConeBeam(
        SOD=float(calib["SOD"]),
        SDD=float(calib["SDD"]),
        NX=args.size,
        NY=args.size,
        NZ=args.size,
        TM=int(recon["rowCount"]),
        TN=int(recon["columnCount"]),
        dd_row=float(recon["ySpacing"]),
        dd_column=float(recon["xSpacing"]),
        voxel_size=args.voxel_size,
        number_of_img=360,
        proj_path=str(args.proj_dir),
        detectorX=float(calib["u0_used"]),
        detectorY=float(calib["v0_used"]),
        pixel_size_raw=float(cal_params["detectorPixelSize"]),
        sx=float(calib["sx"]),
        sy=float(calib["sy"]),
        eta=float(calib["eta"]),
        vc=float(calib["vc_recon"]),
        vs=float(calib["vs_recon"]),
        rotation=float(calib["detector_roll_deg"]),
        angle_offset_deg=float(calib.get("angle_offset_deg", 0.0)),
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        vol_center_x=args.roi_x,
        vol_center_y=args.roi_y,
        vol_center_z=args.roi_z,
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
    print(f"[Done] {nii_path} shape={rec.shape} cuda={use_cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
