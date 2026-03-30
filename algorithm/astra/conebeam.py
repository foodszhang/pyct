import cv2
import numpy as np
import astra as ast
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import sys


class ConeBeam:
    def __init__(
        self,
        *,
        SOD: float,
        SDD: float,
        NX: int,
        NY: int,
        NZ: int,
        TM: int,
        TN: int,
        dd_column: float,
        dd_row: float,
        voxel_size: float,
        number_of_img: int,
        proj_path: str,
        detectorX: float,
        detectorY: float,
        useHu: bool,
        rescale_slope: float,
        rescale_intercept: float,
        pixel_size_raw: float = 0.0748,
        sx: float = 0.5,
        sy: float = 0.5,
    ):
        self.SOD = SOD
        self.SDD = SDD
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.TM = TM
        self.TN = TN
        self.dd_x_raw = dd_column
        self.dd_y_raw = dd_row
        self.voxel_size = voxel_size
        self.pixel_size_raw = pixel_size_raw
        self.sx = sx
        self.sy = sy
        self.proj_path = proj_path
        self.number_of_img = number_of_img
        self.detectorX_raw = detectorX
        self.detectorY_raw = detectorY
        self.vol_geom = ast.create_vol_geom(NX, NY, NZ)
        self.rec_id = ast.data3d.create("-vol", self.vol_geom)
        self.ThreadPoolExecutor = ThreadPoolExecutor(max_workers=20)
        self.w = 0
        self.h = 0
        self.data_lock = Lock()
        self.use_hu = useHu
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        self.I0 = 65535.0
        self.eta = 0.0
        print(f"[Geometry] eta = {self.eta}")

    def load_from_dict(self, img_dict):
        self.data = np.zeros((self.TM, len(img_dict), self.TN), dtype=np.float32)
        img_list = list(img_dict.items())
        img_list = sorted(img_list, key=lambda x: x[0])

        self.detectorX_recon = self.detectorX_raw * self.sx
        self.detectorY_recon = self.detectorY_raw * self.sy
        self.dd_x_recon = self.pixel_size_raw / self.sx
        self.dd_y_recon = self.pixel_size_raw / self.sy

        TM, TN = 0, 0
        for n, v in enumerate(img_list):
            i, img = v
            simg = img.astype(np.float32)
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg, (self.TN, self.TM))
            reshaped = np.clip(reshaped, 1.0, self.I0)
            reshaped = -np.log(reshaped / self.I0)
            self.data[:, n, :] = reshaped
        angles = list(img_dict.keys())
        perAngle = 2 * np.pi / self.number_of_img
        angles = [i * perAngle for i in angles]
        vectors = self.build_cone_vec(
            angles,
            self.SOD,
            self.SDD,
            self.detectorX_recon,
            self.detectorY_recon,
            eta=self.eta,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)

    def build_cone_vec(self, angles, SOD, SDD, u0, v0, eta=0.0):
        """
        构建 ASTRA cone_vec 几何矩阵。

        参数（全部为物理量，单位 mm 或 弧度）：
        - angles : ndarray, 每张投影对应的旋转角 (rad)
        - SOD    : float, 源到旋转中心距离 (mm)
        - SDD    : float, 源到探测器距离 (mm)
        - u0     : float, 光轴打到探测器的水平像素坐标（缩放后图像坐标系）
        - v0     : float, 光轴打到探测器的竖直像素坐标（缩放后图像坐标系）
        - eta    : float, 探测器倾斜参数（无量纲）
        """
        ODD = SDD - SOD

        du = self.pixel_size_raw / self.sx
        dv = self.pixel_size_raw / self.sy

        n_angles = len(angles)
        vectors = np.zeros((n_angles, 12))

        for i, phi in enumerate(angles):
            sp = np.sin(phi)
            cp = np.cos(phi)

            srcX = sp * SOD / self.voxel_size
            srcY = -cp * SOD / self.voxel_size
            srcZ = 0.0

            uX = cp * du / self.voxel_size
            uY = sp * du / self.voxel_size
            uZ = 0.0

            vX = -eta * sp * dv / self.voxel_size
            vY = eta * cp * dv / self.voxel_size
            vZ = -1.0 * dv / self.voxel_size

            dX_on_axis = -sp * ODD / self.voxel_size
            dY_on_axis = cp * ODD / self.voxel_size
            dZ_on_axis = 0.0

            shift_u_pix = self.TN / 2.0 - u0
            shift_v_pix = self.TM / 2.0 - v0

            dX = dX_on_axis + shift_u_pix * uX + shift_v_pix * vX
            dY = dY_on_axis + shift_u_pix * uY + shift_v_pix * vY
            dZ = dZ_on_axis + shift_u_pix * uZ + shift_v_pix * vZ

            vectors[i] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

        return vectors

    def load_img_thread(self, i, number):
        full_path = os.path.join(self.proj_path, f"{i}.tif")
        TM, TN = 0, 0
        if os.path.exists(full_path):
            img = cv2.imread(full_path, -1)
            simg = img.astype(np.float32)
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg, (self.TN, self.TM))
            reshaped = np.clip(reshaped, 1.0, self.I0)
            reshaped = -np.log(reshaped / self.I0)
            self.data_lock.acquire()
            self.data[:, number, :] = reshaped
            self.data_lock.release()
        else:
            print(f"{full_path} not exists")

    def load_img(self):
        futs = []
        count = 0
        numbers = []
        self.angles = []
        for i in range(self.number_of_img):
            full_path = os.path.join(self.proj_path, f"{i}.tif")
            if os.path.exists(full_path):
                count += 1
                numbers.append(i)
        self.data = np.zeros((self.TM, count, self.TN), dtype=np.float32)

        self.detectorX_recon = self.detectorX_raw * self.sx
        self.detectorY_recon = self.detectorY_raw * self.sy
        self.dd_x_recon = self.pixel_size_raw / self.sx
        self.dd_y_recon = self.pixel_size_raw / self.sy

        for n, i in enumerate(numbers):
            futs.append(self.ThreadPoolExecutor.submit(self.load_img_thread, i, n))
        for fut in futs:
            fut.result()
        self.angles = [i * 2 * np.pi / self.number_of_img for i in numbers]

        vectors = self.build_cone_vec(
            self.angles,
            self.SOD,
            self.SDD,
            self.detectorX_recon,
            self.detectorY_recon,
            eta=self.eta,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

        print("[Geometry] Using cone_vec geometry (no postalignment)")
        print(f"[Geometry] SOD = {self.SOD}, SDD = {self.SDD}")
        print(f"[Geometry] u0 = {self.detectorX_recon}, v0 = {self.detectorY_recon}")
        print(f"[Geometry] eta = {self.eta}")

        print("[Preprocess] Using Beer-Lambert projection: -log(I/I0)")
        print(f"[Preprocess] I0 = {self.I0}")
        print(f"[Preprocess] data shape = {self.data.shape}")
        print(f"[Preprocess] data min = {self.data.min():.6f}")
        print(f"[Preprocess] data max = {self.data.max():.6f}")
        print(f"[Preprocess] data mean = {self.data.mean():.6f}")

    def reconstruct(self):
        # self.cfg_fdk = ast.astra_dict('SIRT3D_CUDA')
        self.cfg_fdk = ast.astra_dict("FDK_CUDA")
        self.cfg_fdk["ProjectionDataId"] = self.proj_id
        self.cfg_fdk["ReconstructionDataId"] = self.rec_id
        alg_id = ast.algorithm.create(self.cfg_fdk)
        ast.algorithm.run(alg_id, 1)
        rec = ast.data3d.get(self.rec_id)
        ar = rec[:]
        ast.algorithm.delete(alg_id)
        ast.data3d.delete(self.rec_id)
        ast.data3d.delete(self.proj_id)
        if self.use_hu:
            print(
                "[HU] useHu=True was requested, but HU conversion is disabled in current debug stage"
            )
        print("[HU] Disabled in current debug stage")
        print("[Reconstruction] Returning raw attenuation reconstruction")
        print(f"[Reconstruction] min = {ar.min():.6f}")
        print(f"[Reconstruction] max = {ar.max():.6f}")
        print(f"[Reconstruction] mean = {ar.mean():.6f}")
        return ar


if __name__ == "__main__":
    import nibabel as nib
    import os

    output_dir = "/home/foods/pro/pyct_old/pyct/recon_output/phaseB/"
    os.makedirs(output_dir, exist_ok=True)

    cb = ConeBeam(
        SOD=885.41,
        TN=768,
        TM=972,
        SDD=850.00,
        NX=512,
        NY=512,
        NZ=512,
        dd_row=0.0748,
        dd_column=0.0748,
        voxel_size=0.25,
        number_of_img=360,
        proj_path=r"/home/foods/pro/data/20260327-jz-1/",
        detectorX=916.88,
        detectorY=1013.91,
        pixel_size_raw=0.0748,
        sx=0.5,
        sy=0.5,
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
    )
    cb.eta = 0.013794
    cb.load_img()
    rec = cb.reconstruct()
    print(rec.shape, rec.dtype)

    print(
        f"[Stats] min={rec.min():.6f}, max={rec.max():.6f}, mean={rec.mean():.6f}, std={rec.std():.6f}"
    )
    rec_pos = rec[rec > 0]
    if len(rec_pos) > 0:
        print(f"[Stats] mean of >0 voxels: {rec_pos.mean():.6f}")

    rec_raw = rec.astype(np.float32)
    rec_scaled = (rec / rec.max() * 1000).astype(np.int16)

    output_dir = "/home/foods/pro/pyct_old/pyct/recon_output/phaseB_joint_sod_sdd/"
    os.makedirs(output_dir, exist_ok=True)

    nii_path_raw = os.path.join(output_dir, "phaseB_joint_sod_sdd_rec_raw.nii.gz")
    nii_img_raw = nib.Nifti1Image(rec_raw, np.eye(4))
    nib.save(nii_img_raw, nii_path_raw)
    print(f"Saved {nii_path_raw}")

    nii_path = os.path.join(output_dir, "phaseB_joint_sod_sdd_rec.nii.gz")
    nii_img = nib.Nifti1Image(rec_scaled, np.eye(4))
    nib.save(nii_img, nii_path)
    print(f"Saved {nii_path}")
