import sys

sys.path.insert(0, "/home/foods/pro/pyct_old/pyct")
from algorithm.astra.conebeam import ConeBeam
import numpy as np
import cv2
import os

calibrated = dict(
    SOD=907.0,
    SDD=970.97,
    u0=929.42,
    v0=987.18,
    theta=0.0376,
    eta=0.0034,
)

cb = ConeBeam(
    SOD=calibrated["SOD"],
    SDD=calibrated["SDD"],
    NX=512,
    NY=512,
    NZ=512,
    TM=1944,
    TN=1536,
    dd_column=0.0748,
    dd_row=0.0748,
    voxel_size=0.2,
    number_of_img=361,
    proj_path="/home/foods/pro/data/20260327-jz-1",
    detectorX=calibrated["u0"],
    detectorY=calibrated["v0"],
    rotation_angle=0.0,
    useHu=False,
    rescale_slope=1.0,
    rescale_intercept=0.0,
    theta=calibrated["theta"],
    eta=calibrated["eta"],
)

print("Loading projections...")
cb.load_img()
print("Reconstructing...")
rec = cb.reconstruct()
print(f"Reconstruction shape: {rec.shape}")

save_dir = "/home/foods/pro/pyct_old/pyct/recon_output"
os.makedirs(save_dir, exist_ok=True)

for i, name in enumerate([" axial (xy)", " coronal (xz)", " sagittal (yz)"]):
    if i == 0:
        slice_img = rec[:, :, rec.shape[2] // 2]
    elif i == 1:
        slice_img = rec[:, rec.shape[1] // 2, :]
    else:
        slice_img = rec[rec.shape[0] // 2, :, :]
    slice_norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    save_path = os.path.join(save_dir, f"slice{i}.png")
    cv2.imwrite(save_path, slice_norm)
    print(f"Saved {name} slice to {save_path}")

print("Done!")
