import cv2
import numpy as np
import astra as ast
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import sys
from math import pi, sin, cos, sqrt


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
        theta: float = 0.0,
        eta: float = 0.0,
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
        self.theta = theta
        self.eta = eta
        half_x = NX * voxel_size / 2.0
        half_y = NY * voxel_size / 2.0
        half_z = NZ * voxel_size / 2.0
        self.vol_geom = ast.create_vol_geom(
            NX,
            NY,
            NZ,
            -half_x,
            half_x,
            -half_y,
            half_y,
            -half_z,
            half_z,
        )
        self.rec_id = ast.data3d.create("-vol", self.vol_geom)
        self.ThreadPoolExecutor = ThreadPoolExecutor(max_workers=20)
        self.w = TN
        self.h = TM
        self.data_lock = Lock()
        self.use_hu = useHu
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        self.du = (self.w / 2.0 - detectorX) * self.TN / self.w
        self.dv = (detectorY - self.h / 2.0) * self.TM / self.h
        self.numbers = []
        self.data = None
        self.proj_geom = None
        self.proj_id = None
        self.cfg_fdk = None
        self.angles = []

    # ── 修改 2：generate_cone_vec ──
    def generate_cone_vec(self):
        SOD = self.SOD
        SDD = self.SDD
        detectorX = self.detectorX
        detectorY = self.detectorY
        theta = self.theta
        eta = self.eta
        dd_u = self.dd_x
        dd_v = self.dd_y
        TM = self.TM
        TN = self.TN
        numbers = self.numbers
        N = len(numbers)
        vectors = np.zeros((N, 12), dtype=np.float64)
        for idx, num_i in enumerate(numbers):
            phi = num_i * 2 * pi / self.number_of_img
            src_x = SOD * sin(phi)
            src_y = -SOD * cos(phi)
            src_z = 0.0
            u_base = np.array([cos(phi), sin(phi), 0.0]) * dd_u
            v_base = np.array([-eta * sin(phi), eta * cos(phi), 1.0])
            v_norm = sqrt(v_base[0] ** 2 + v_base[1] ** 2 + v_base[2] ** 2)
            v_base = v_base / v_norm * dd_v
            u_vec = cos(theta) * u_base + sin(theta) * v_base
            v_vec = -sin(theta) * u_base + cos(theta) * v_base
            det_ideal_x = -(SDD - SOD) * sin(phi)
            det_ideal_y = (SDD - SOD) * cos(phi)
            det_ideal_z = 0.0
            u0_offset_mm = (detectorX - TN / 2.0) * dd_u
            v0_offset_mm = (detectorY - TM / 2.0) * dd_v
            u_hat = u_vec / dd_u
            v_hat = v_vec / dd_v
            det_x = det_ideal_x + u0_offset_mm * u_hat[0] + v0_offset_mm * v_hat[0]
            det_y = det_ideal_y + u0_offset_mm * u_hat[1] + v0_offset_mm * v_hat[1]
            det_z = det_ideal_z + u0_offset_mm * u_hat[2] + v0_offset_mm * v_hat[2]
            vectors[idx] = [
                src_x,
                src_y,
                src_z,
                det_x,
                det_y,
                det_z,
                u_vec[0],
                u_vec[1],
                u_vec[2],
                v_vec[0],
                v_vec[1],
                v_vec[2],
            ]
        return vectors

    # END generate_cone_vec

    # ── 修改 4：load_from_dict ──
    def load_from_dict(self, img_dict):
        self.data = np.zeros((self.TM, len(img_dict), self.TN), dtype=np.float32)
        img_list = list(img_dict.items())
        img_list = sorted(img_list, key=lambda x: x[0])
        for n, v in enumerate(img_list):
            i, img = v
            simg = img
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg.astype(np.float32), (self.TN, self.TM))
            reshaped = np.clip(reshaped, 1.0, 65535.0)
            reshaped = -np.log(reshaped / 65535.0)
            self.data[:, n, :] = reshaped
        numbers = [v[0] for v in img_list]
        self.numbers = numbers
        vectors = self.generate_cone_vec()
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)

    # END load_from_dict

    def load_img_thread(self, i, number):
        full_path = os.path.join(self.proj_path, f"{i}.tif")
        TM, TN = 0, 0
        if os.path.exists(full_path):
            img = cv2.imread(full_path, -1)
            simg = img
            TM, TN = simg.shape
            self.w = TN
            self.h = TM
            reshaped = cv2.resize(simg.astype(np.float32), (self.TN, self.TM))
            reshaped = np.clip(reshaped, 1.0, 65535.0)
            reshaped = -np.log(reshaped / 65535.0)
            self.data_lock.acquire()
            self.data[:, number, :] = reshaped
            self.data_lock.release()
        else:
            print(f"{full_path} not exists")

    # ── 修改 3：load_img ──
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
        self.numbers = numbers
        self.data = np.zeros((self.TM, count, self.TN), dtype=np.float32)
        for n, i in enumerate(numbers):
            futs.append(self.ThreadPoolExecutor.submit(self.load_img_thread, i, n))
        for fut in futs:
            fut.result()
        vectors = self.generate_cone_vec()
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

    # END load_img

    def reconstruct(self):
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/foods/pro/data/20260327-jz-1")
    args = parser.parse_args()

    NX = NY = NZ = 512
    TM = 1944
    TN = 1536
    voxel_size = 0.2

    calibrated = dict(
        SOD=907.0, SDD=970.97, u0=910.64, v0=987.17, theta=0.003917, eta=0.0
    )

    cb = ConeBeam(
        SOD=calibrated["SOD"],
        SDD=calibrated["SDD"],
        NX=NX,
        NY=NY,
        NZ=NZ,
        TM=TM,
        TN=TN,
        dd_column=0.0748,
        dd_row=0.0748,
        voxel_size=voxel_size,
        number_of_img=361,
        proj_path=args.data,
        detectorX=calibrated["u0"],
        detectorY=calibrated["v0"],
        rotation_angle=0.0,
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
        theta=calibrated["theta"],
        eta=calibrated["eta"],
    )

    print("=== 校准参数 ===")
    print(f"  SOD={cb.SOD}, SDD={cb.SDD}")
    print(f"  detectorX={cb.detectorX}, detectorY={cb.detectorY}")
    print(f"  theta={cb.theta}, eta={cb.eta}")
    print(f"  voxel_size={voxel_size}, NX=NY=NZ={NX}")
    print(f"  number_of_img={cb.number_of_img}")

    cb.load_img()
    print(f"\nLoaded {len(cb.numbers)} projections")
    print(
        f"data shape: {cb.data.shape}, min={cb.data.min():.4f}, max={cb.data.max():.4f}"
    )

    rec = cb.reconstruct()
    print(f"\n=== 重建结果 ===")
    print(f"  rec.shape={rec.shape}, dtype={rec.dtype}")
    print(f"  rec.min={rec.min():.4f}, max={rec.max():.4f}, mean={rec.mean():.4f}")

    NZ, NY, NX = rec.shape

    bead_z_mm = [0, 9, 19, 29, 38, 48]
    bead_z_indices = [round((b_z / voxel_size) + NZ / 2) for b_z in bead_z_mm]
    print(f"\n=== 钢珠 Z 索引 ===")
    print(f"  bead_z_mm={bead_z_mm}")
    print(f"  bead_z_indices={bead_z_indices}")

    print(f"\n=== Axial 钢珠位置切片 ===")
    for k, z_idx in enumerate(bead_z_indices):
        if 0 <= z_idx < NZ:
            slice_img = rec[z_idx, :, :]
            mn = slice_img.min()
            mx = slice_img.max()
            mean = slice_img.mean()
            std = slice_img.std()
            save_path = f"/tmp/axial_ball_{k}.png"
            norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            cv2.imwrite(save_path, norm)
            print(f"  ball {k} z={z_idx}: mean={mean:.4f}, max={mx:.4f} -> {save_path}")
        else:
            print(f"  ball {k} z={z_idx}: OUT OF RANGE")

    print(f"\n=== 标准三视图 ===")
    # axial (top-down)
    axial = rec[NZ // 2, :, :]
    norm_axial = cv2.normalize(axial, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("/tmp/slice_axial.png", norm_axial)
    print(f"  axial (z={NZ // 2}): mean={axial.mean():.4f}, std={axial.std():.4f}")

    # coronal (front)
    coronal = rec[:, NY // 2, :]
    norm_coronal = cv2.normalize(coronal, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    cv2.imwrite("/tmp/slice_coronal.png", norm_coronal)
    print(
        f"  coronal (y={NY // 2}): mean={coronal.mean():.4f}, std={coronal.std():.4f}"
    )

    # sagittal (side)
    sagittal = rec[:, :, NX // 2]
    norm_sagittal = cv2.normalize(sagittal, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    cv2.imwrite("/tmp/slice_sagittal.png", norm_sagittal)
    print(
        f"  sagittal (x={NX // 2}): mean={sagittal.mean():.4f}, std={sagittal.std():.4f}"
    )

    print("\nDone! Slices saved to /tmp/")
