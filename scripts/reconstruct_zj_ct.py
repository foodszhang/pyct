import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.astra.conebeam import ConeBeam


def numeric_tif_count(ct_dir: Path) -> int:
    count = 0
    for path in ct_dir.glob("*.tif"):
        try:
            float(path.stem)
        except ValueError:
            continue
        count += 1
    return count


def reconstruct_one(
    ct_dir: Path,
    out_name: str,
    overwrite: bool,
    params: dict,
    algorithm: str,
    iterations: int,
    filter_type: str,
    ring_correction: bool,
    ring_kernel_size: int,
    drop_duplicate_360: bool,
    fill_missing: bool,
) -> None:
    out_path = ct_dir.parent / out_name
    if out_path.exists() and not overwrite:
        print(f"[Skip] {out_path} exists")
        return

    count = numeric_tif_count(ct_dir)
    if count == 0:
        print(f"[Skip] no numeric tif projections: {ct_dir}")
        return

    print(f"[Start] {ct_dir} projections={count}")
    cb = ConeBeam(
        SOD=params["SOD"],
        SDD=params["SDD"],
        NX=params["NX"],
        NY=params["NY"],
        NZ=params["NZ"],
        TM=params["TM"],
        TN=params["TN"],
        dd_row=params["dd_row"],
        dd_column=params["dd_column"],
        voxel_size=params["voxel_size"],
        number_of_img=360,
        proj_path=str(ct_dir),
        detectorX=params["detector_x"],
        detectorY=params["detector_y"],
        pixel_size_raw=params["pixel_size_raw"],
        sx=params["sx"],
        sy=params["sy"],
        eta=params["eta"],
        vc=params["vc"],
        vs=params["vs"],
        rotation=params["rotation"],
        angle_offset_deg=params["angle_offset_deg"],
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        vol_center_x=params["vol_center_x"],
        vol_center_y=params["vol_center_y"],
        vol_center_z=params["vol_center_z"],
        air_percentile=params.get("air_percentile"),
        air_zero_percentile=params.get("air_zero_percentile"),
        mass_normalize=params.get("mass_normalize", False),
    )
    cb.load_img(
        angle_from_filename=True,
        drop_duplicate_360=drop_duplicate_360,
        fill_missing_degrees=fill_missing,
    )
    rec, use_cuda = cb.reconstruct(
        filter_type=filter_type,
        algorithm=algorithm,
        iterations=iterations,
        non_neg_constraint=True,
        ring_correction=ring_correction,
        ring_kernel_size=ring_kernel_size,
    )
    nib.save(nib.Nifti1Image(rec.astype(np.float32, copy=False), np.eye(4)), out_path)
    print(f"[Done] {out_path} shape={rec.shape} cuda={use_cuda}")


def iter_ct_dirs(root: Path, samples: list[str]) -> list[Path]:
    if samples:
        return [root / sample / "ct" for sample in samples]
    dirs = []
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit() and (path / "ct").is_dir():
            dirs.append(path / "ct")
    return sorted(dirs, key=lambda p: int(p.parent.name))


def params_from_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {
            "SOD": 908.8,
            "SDD": 959.6,
            "NX": 512,
            "NY": 512,
            "NZ": 512,
            "TM": 972,
            "TN": 768,
            "dd_row": 0.1496,
            "dd_column": 0.1496,
            "voxel_size": 0.25,
            "detector_x": 916.88,
            "detector_y": 1013.91,
            "pixel_size_raw": 0.0748,
            "sx": 0.5,
            "sy": 0.5,
            "eta": 0.0,
            "vc": -5.691,
            "vs": -7.434,
            "rotation": 0.0,
            "angle_offset_deg": 0.0,
            "vol_center_x": 0.0,
            "vol_center_y": 0.0,
            "vol_center_z": 0.0,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    recon = cfg["ReconParam"]
    calib = cfg.get("CalibResult", {})
    cal_param = cfg.get("CalibrationParam", {})
    return {
        "SOD": float(recon["SOD"]),
        "SDD": float(recon["SDD"]),
        "NX": int(recon["voxelSizeX"]),
        "NY": int(recon["voxelSizeY"]),
        "NZ": int(recon["voxelSizeZ"]),
        "TM": int(recon["rowCount"]),
        "TN": int(recon["columnCount"]),
        "dd_row": float(recon["ySpacing"]),
        "dd_column": float(recon["xSpacing"]),
        "voxel_size": float(recon["voxelPixelSize"]),
        "detector_x": float(recon["detectorX"]),
        "detector_y": float(recon["detectorY"]),
        "pixel_size_raw": float(cal_param.get("detectorPixelSize", 0.0748)),
        "sx": float(calib.get("sx", 0.5)),
        "sy": float(calib.get("sy", 0.5)),
        "eta": float(calib.get("eta", 0.0)),
        "vc": float(calib.get("vc_recon", 0.0)),
        "vs": float(calib.get("vs_recon", 0.0)),
        "rotation": float(calib.get("detector_roll_deg", recon.get("rotation", 0.0))),
        "angle_offset_deg": float(calib.get("angle_offset_deg", 0.0)),
        "vol_center_x": float(recon.get("roiCenterX", 0.0)),
        "vol_center_y": float(recon.get("roiCenterY", 0.0)),
        "vol_center_z": float(recon.get("roiCenterZ", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/foods/pro/true_data/ZJ")
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--out-name", default="rec_pyct_calibrated.nii.gz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--sod", type=float)
    parser.add_argument("--sdd", type=float)
    parser.add_argument("--detector-x", type=float)
    parser.add_argument("--detector-y", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--vc", type=float)
    parser.add_argument("--vs", type=float)
    parser.add_argument("--roi-x", type=float)
    parser.add_argument("--roi-y", type=float)
    parser.add_argument("--roi-z", type=float)
    parser.add_argument("--algorithm", default="FDK", choices=["FDK", "SIRT", "CGLS"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--filter-type", default="Ram-Lak")
    parser.add_argument("--ring-correction", action="store_true")
    parser.add_argument("--ring-kernel-size", type=int, default=9)
    parser.add_argument("--keep-360", action="store_true")
    parser.add_argument("--fill-missing", action="store_true")
    parser.add_argument("--air-percentile", type=float)
    parser.add_argument("--air-zero-percentile", type=float)
    parser.add_argument("--mass-normalize", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    params = params_from_config(args.config)
    if args.size is not None:
        params["NX"] = args.size
        params["NY"] = args.size
        params["NZ"] = args.size
    if args.voxel_size is not None:
        params["voxel_size"] = args.voxel_size
    if args.sod is not None:
        params["SOD"] = args.sod
    if args.sdd is not None:
        params["SDD"] = args.sdd
    if args.detector_x is not None:
        params["detector_x"] = args.detector_x
    if args.detector_y is not None:
        params["detector_y"] = args.detector_y
    if args.eta is not None:
        params["eta"] = args.eta
    if args.vc is not None:
        params["vc"] = args.vc
    if args.vs is not None:
        params["vs"] = args.vs
    if args.roi_x is not None:
        params["vol_center_x"] = args.roi_x
    if args.roi_y is not None:
        params["vol_center_y"] = args.roi_y
    if args.roi_z is not None:
        params["vol_center_z"] = args.roi_z
    if args.air_percentile is not None:
        params["air_percentile"] = args.air_percentile
    if args.air_zero_percentile is not None:
        params["air_zero_percentile"] = args.air_zero_percentile
    if args.mass_normalize:
        params["mass_normalize"] = True
    print(f"[ConfigParams] {params}")
    for ct_dir in iter_ct_dirs(root, args.samples):
        reconstruct_one(
            ct_dir,
            args.out_name,
            args.overwrite,
            params,
            args.algorithm,
            args.iterations,
            args.filter_type,
            args.ring_correction,
            args.ring_kernel_size,
            not args.keep_360,
            args.fill_missing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
