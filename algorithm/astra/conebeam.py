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
        rotation_angle: float,
        useHu: bool,
        rescale_slope: float,
        rescale_intercept: float,
    ):
        self.SOD = SOD
        self.SDD = SDD
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.TM = TM
        self.TN = TN
        self.dd_x = dd_column
        self.dd_y = dd_row
        self.dd = voxel_size
        self.proj_path = proj_path
        self.number_of_img = number_of_img
        self.detectorX = detectorX
        self.detectorY = detectorY
        self.rotation_angle = rotation_angle
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
        TM, TN = 0, 0
        for n, v in enumerate(img_list):
            i, img = v
            simg = img
            rotation_mat = cv2.getRotationMatrix2D(
                (self.detectorX, self.detectorY), self.rotation_angle, 1
            )
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            max_img = np.max(simg)
            reshaped = cv2.warpAffine(
                simg, rotation_mat, (TN, TM), borderValue=float(max_img))
            reshaped = cv2.resize(reshaped, (self.TN, self.TM))
            self.data[:, n, :] = reshaped
        angles = list(img_dict.keys())
        perAngle = 2 * np.pi / self.number_of_img
        angles = [i * perAngle for i in angles]
        self.proj_geom = ast.create_proj_geom(
            "cone",
            self.dd_y / self.dd,
            self.dd_x / self.dd,
            self.TM,
            self.TN,
            angles,
            self.SOD / self.dd,
            (self.SDD - self.SOD) / self.dd,
        )
        center  = TN / 2, TM/ 2

        self.proj_geom = ast.geom_postalignment(self.proj_geom, (self.du, self.dv))

    def load_img_thread(self, i, number):
        full_path = os.path.join(self.proj_path, f"{i}.tif")
        TM, TN = 0, 0
        if os.path.exists(full_path):
            img = cv2.imread(full_path, -1)
            simg = img
            # 这里旋转角度是负数是实验的得到的有待进一步验证
            rotation_mat = cv2.getRotationMatrix2D(
                (self.detectorX, self.detectorY), -self.rotation_angle, 1
            )
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            max_img = np.max(simg)
            reshaped = cv2.warpAffine(
                simg, rotation_mat, (TN, TM), borderValue=float(max_img))
            reshaped = cv2.resize(reshaped, (self.TN, self.TM))
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
                count+=1
                numbers.append(i)
        self.data = np.zeros((self.TM, count, self.TN), dtype=np.float32)


        for n, i in enumerate(numbers):
            futs.append(self.ThreadPoolExecutor.submit(self.load_img_thread, i, n))
        for fut in futs:
            fut.result()
        self.angles = [i * 2 * np.pi / self.number_of_img for i in numbers]

        self.proj_geom = ast.create_proj_geom(
            "cone",
            self.dd_y / self.dd,
            self.dd_x / self.dd,
            self.TM,
            self.TN,
            self.angles,
            self.SOD / self.dd,
            (self.SDD - self.SOD) / self.dd,
        )
        # 这里du，dv是实验得到的， astra的具体坐标系不太清楚
        du = (self.w/2-self.detectorX) * self.TN /self.w
        dv = (self.detectorY - self.h / 2) * self.TM / self.h
        self.proj_geom = ast.geom_postalignment(self.proj_geom, (du, dv))
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

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
    N = 512
    TM = 1944
    TN = 1536
    # cb = ConeBeam(SOD=722.679, TN=512, TM=512,  SDD=778.676, N = 512, dd_row=0.2840,dd_column=0.2244, voxel_size=0.2, number_of_img=360, proj_path=r"../../data/alway1", du=du, dv=-dv, dark_file='../../data/dark.tif', empty_file='../../data/empty.tif')
    cb = ConeBeam(
        SOD=910,
        TN=512,
        TM=512,
        SDD=979.239,
        NX=512,
        NY=512,
        NZ=512,
        dd_row=0.2840,
        dd_column=0.2244,
        voxel_size=0.22,
        number_of_img=360,
        proj_path=r"../../data/123-2/",
        detectorX=750,
        detectorY=1013,
        rotation_angle=-0.35,
    )
    cb.load_img()
    rec = cb.reconstruct()
    print(rec.shape, rec.dtype)
