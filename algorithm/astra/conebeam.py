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
            simg = img
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg, (self.TN, self.TM))
            self.data[:, n, :] = reshaped
        angles = list(img_dict.keys())
        perAngle = 2 * np.pi / self.number_of_img
        angles = [i * perAngle for i in angles]
        self.proj_geom = ast.create_proj_geom(
            "cone",
            self.dd_y_recon / self.voxel_size,
            self.dd_x_recon / self.voxel_size,
            self.TM,
            self.TN,
            angles,
            self.SOD / self.voxel_size,
            (self.SDD - self.SOD) / self.voxel_size,
        )
        du = self.TN / 2 - self.detectorX_recon
        dv = self.detectorY_recon - self.TM / 2
        self.proj_geom = ast.geom_postalignment(self.proj_geom, (du, dv))

    def load_img_thread(self, i, number):
        full_path = os.path.join(self.proj_path, f"{i}.tif")
        TM, TN = 0, 0
        if os.path.exists(full_path):
            img = cv2.imread(full_path, -1)
            simg = img
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg, (self.TN, self.TM))
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

        self.proj_geom = ast.create_proj_geom(
            "cone",
            self.dd_y_recon / self.voxel_size,
            self.dd_x_recon / self.voxel_size,
            self.TM,
            self.TN,
            self.angles,
            self.SOD / self.voxel_size,
            (self.SDD - self.SOD) / self.voxel_size,
        )
        du = self.TN / 2 - self.detectorX_recon
        dv = self.detectorY_recon - self.TM / 2
        self.proj_geom = ast.geom_postalignment(self.proj_geom, (du, dv))
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

        print(f"[Phase B] 768x972 scheme, NO rotation_angle:")
        print(
            f"  detectorX_recon = {self.detectorX_recon:.3f}, detectorY_recon = {self.detectorY_recon:.3f}"
        )
        print(
            f"  dd_x_recon = {self.dd_x_recon:.4f}, dd_y_recon = {self.dd_y_recon:.4f}"
        )
        print(f"  du = {du:.3f}, dv = {dv:.3f}")

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
        print("finish rec", self.use_hu)
        if self.use_hu:
            ar = self.rescale_slope * ar + self.rescale_intercept
        return ar


if __name__ == "__main__":
    import nibabel as nib

    cb = ConeBeam(
        SOD=910.7,
        TN=768,
        TM=972,
        SDD=978.11,
        NX=512,
        NY=512,
        NZ=512,
        dd_row=0.0748,
        dd_column=0.0748,
        voxel_size=0.25,
        number_of_img=360,
        proj_path=r"/home/foods/pro/data/20260327-jz-1/",
        detectorX=751.77,
        detectorY=1013.91,
        pixel_size_raw=0.0748,
        sx=0.5,
        sy=0.5,
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
    )
    cb.load_img()
    rec = cb.reconstruct()
    print(rec.shape, rec.dtype)

    # Save volume
    nii_img = nib.Nifti1Image(rec, np.eye(4))
    nib.save(nii_img, "phaseB_rec.nii.gz")
    print("Saved phaseB_rec.nii.gz")
