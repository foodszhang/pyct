import cv2
import numpy as np
import astra as ast
import nibabel as nib

sid = 722.6798  # source to isocenter distance
sdd = 778.6765  # source to detector distanc1e

N = 512
TM = 1944
TN = 1536
du = 6.3708
dv = -158.6277

dd = 74.8 / 1000
# data = np.zeros((TM,360,TN), dtype=np.float32)
data = np.zeros((N, 360, N), dtype=np.float32)
import os

for i in range(360):
    if os.path.exists(f"../../data/alway/{i}.tif"):
        img = cv2.imread(f"../../data/alway/{i}.tif", -1)
        reshaped = cv2.resize(img, (N, N))
        print(f"{i}.tif")
        # if os.path.exists(f"../../data/1206-360-outer/{i}.tif"):
        #    print(f"{i}.tif")
        #    img = cv2.imread(f"../../data/1206-360-outer/{i}.tif", -1)
        #    # X,angles,Y

        data[:, i, :] = reshaped


vol_geom = ast.create_vol_geom(N, N, N)
# proj_geom = astra_create_proj_geom('cone',  det_spacing_x, det_spacing_y, det_row_count, det_col_count, angles, source_origin, origin_det);
# det_spacing_x = det_spacing_y = 0.0748 像素间距
# det_row_count = 1944 探测器行数
# det_col_count = 1536 探测器列数
# angles 以弧度为单位的投影
angles = np.linspace(0, 2 * np.pi, 360, False)
proj_geom = ast.create_proj_geom(
    "cone", 0.4 * TN / N, 0.4 * TM / N, N, N, angles, sid / dd, (sdd - sid) / dd
)
# proj_geom = ast.create_proj_geom('cone', 0.4 , 0.4  , TM, TN, angles, sid/dd, (sdd-sid)/dd)
# porj_geom = ast.geom_postalignment(proj_geom, du, dv)
rec_id = ast.data3d.create("-vol", vol_geom)
proj_id = ast.data3d.create("-proj3d", proj_geom, data)
cfg_fdk = ast.astra_dict("FDK_CUDA")
cfg_fdk["ProjectionDataId"] = proj_id
cfg_fdk["ReconstructionDataId"] = rec_id
alg_id = ast.algorithm.create(cfg_fdk)

# cfg_fdk = ast.astra_dict('CGLS3D_CUDA')
# cfg_fdk['ProjectionDataId'] = proj_id
# cfg_fdk['ReconstructionDataId'] = rec_id
# alg_id = ast.algorithm.create(cfg_fdk)

ast.algorithm.run(alg_id, 1)
rec = ast.data3d.get(rec_id)


nii_img = nib.Nifti1Image(rec, np.eye(4))
nib.save(nii_img, "rec.nii.gz")
print(rec.shape, rec.dtype)

ast.algorithm.delete(alg_id)
ast.data3d.delete(rec_id)
ast.data3d.delete(proj_id)
