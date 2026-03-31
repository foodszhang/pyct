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
        eta: float = 0.0,
        vc: float = 0.0,
        vs: float = 0.0,
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
        self.eta = eta
        self.vc = vc
        self.vs = vs
        print(f"[Geometry] eta = {self.eta}, vc = {self.vc}, vs = {self.vs}")

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

        if self.w > 0:
            expected_tn = int(self.w * self.sx) if self.w > 0 else self.TN
            expected_tm = int(self.h * self.sy) if self.h > 0 else self.TM
            if abs(self.TN - expected_tn) > 2 or abs(self.TM - expected_tm) > 2:
                print(f"[Warn] TN/TM 与探测器尺寸*sx/sy 不匹配!")
                print(f"[Warn]   原始图像: {self.w}x{self.h}")
                print(f"[Warn]   当前 TN={self.TN}, TM={self.TM}")
                print(f"[Warn]   期望 TN={expected_tn}, TM={expected_tm} (raw * sx/sy)")
                print(f"[Warn]   这会导致重建几何不正确!")

        if self.detectorX_recon > self.TN or self.detectorY_recon > self.TM:
            print(f"[Warn] 探测器中心超出图像范围!")
            print(f"[Warn]   u0_recon={self.detectorX_recon}, TN={self.TN}")
            print(f"[Warn]   v0_recon={self.detectorY_recon}, TM={self.TM}")

        vectors = self.build_cone_vec(
            angles,
            self.SOD,
            self.SDD,
            self.detectorX_recon,
            self.detectorY_recon,
            eta=self.eta,
            vc=self.vc,
            vs=self.vs,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)

    def build_cone_vec(self, angles, SOD, SDD, u0, v0, eta=0.0, vc=0.0, vs=0.0):
        """
        构建 ASTRA cone_vec 几何矩阵。

        参数（全部为物理量，单位 mm 或 弧度）：
        - angles : ndarray, 每张投影对应的旋转角 (rad)
        - SOD    : float, 源到旋转中心距离 (mm)
        - SDD    : float, 源到探测器距离 (mm)
        - u0     : float, 光轴打到探测器的水平像素坐标（缩放后图像坐标系）
        - v0     : float, 光轴打到探测器的竖直像素坐标（缩放后图像坐标系）
        - eta    : float, 探测器倾斜参数（无量纲）
        - vc     : float, v-shift cosine coefficient (recon pixels)
        - vs     : float, v-shift sine coefficient (recon pixels)
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
            v_shift = -(vc * cp + vs * sp)
            shift_v_pix = self.TM / 2.0 - v0 + v_shift

            dX = dX_on_axis + shift_u_pix * uX + shift_v_pix * vX
            dY = dY_on_axis + shift_u_pix * uY + shift_v_pix * vY
            dZ = dZ_on_axis + shift_u_pix * uZ + shift_v_pix * vZ

            vectors[i] = [srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ]

        return vectors

    def load_img_thread(self, filename, number):
        full_path = os.path.join(self.proj_path, filename)
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

    def load_img(self, angle_from_filename: bool = False, progress_callback=None):
        if angle_from_filename:
            tif_files = [f for f in os.listdir(self.proj_path) if f.endswith(".tif")]
            parsed = []
            for f in tif_files:
                try:
                    angle_deg = float(f.replace(".tif", ""))
                    parsed.append((f, angle_deg))
                except ValueError:
                    continue
            if len(parsed) == 0:
                raise ValueError(
                    f"[Angles] No valid .tif files found in {self.proj_path}"
                )
            parsed.sort(key=lambda x: x[1])
            filenames = [p[0] for p in parsed]
            angle_deg_list = [p[1] for p in parsed]
            angles = [a * np.pi / 180.0 for a in angle_deg_list]
            count = len(filenames)
            print(f"[Angles] 从文件名加载 {count} 张投影（支持非均匀/缺角）")
            print(
                f"[Angles] 角度范围: {min(angle_deg_list):.1f}..{max(angle_deg_list):.1f} deg"
            )
            print(f"[Angles] 前5个角度: {angle_deg_list[:5]}")
        else:
            filenames = []
            for i in range(self.number_of_img):
                full_path = os.path.join(self.proj_path, f"{i}.tif")
                if os.path.exists(full_path):
                    filenames.append(f"{i}.tif")
            count = len(filenames)
            angles = [i * 2 * np.pi / self.number_of_img for i in range(count)]
            print(f"[Angles] 从序号加载 {count} 张投影（均匀假设）")

        self.data = np.zeros((self.TM, count, self.TN), dtype=np.float32)

        self.detectorX_recon = self.detectorX_raw * self.sx
        self.detectorY_recon = self.detectorY_raw * self.sy
        self.dd_x_recon = self.pixel_size_raw / self.sx
        self.dd_y_recon = self.pixel_size_raw / self.sy

        _counter = [0]
        _counter_lock = Lock()

        def _on_future_done(fut):
            with _counter_lock:
                _counter[0] += 1
                if progress_callback is not None:
                    progress_callback(_counter[0], count, "加载投影")

        futs = []
        for n, fname in enumerate(filenames):
            f = self.ThreadPoolExecutor.submit(self.load_img_thread, fname, n)
            f.add_done_callback(_on_future_done)
            futs.append(f)
        for fut in futs:
            fut.result()
        self.angles = angles

        if self.w > 0:
            expected_tn = int(self.w * self.sx) if self.w > 0 else self.TN
            expected_tm = int(self.h * self.sy) if self.h > 0 else self.TM
            if abs(self.TN - expected_tn) > 2 or abs(self.TM - expected_tm) > 2:
                print(f"[Warn] TN/TM 与探测器尺寸*sx/sy 不匹配!")
                print(f"[Warn]   原始图像: {self.w}x{self.h}")
                print(f"[Warn]   当前 TN={self.TN}, TM={self.TM}")
                print(f"[Warn]   期望 TN={expected_tn}, TM={expected_tm} (raw * sx/sy)")
                print(f"[Warn]   这会导致重建几何不正确!")

        if self.detectorX_recon > self.TN or self.detectorY_recon > self.TM:
            print(f"[Warn] 探测器中心超出图像范围!")
            print(f"[Warn]   u0_recon={self.detectorX_recon}, TN={self.TN}")
            print(f"[Warn]   v0_recon={self.detectorY_recon}, TM={self.TM}")

        vectors = self.build_cone_vec(
            self.angles,
            self.SOD,
            self.SDD,
            self.detectorX_recon,
            self.detectorY_recon,
            eta=self.eta,
            vc=self.vc,
            vs=self.vs,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

        print("[Geometry] Using cone_vec geometry (no postalignment)")
        print(f"[Geometry] SOD = {self.SOD}, SDD = {self.SDD}")
        print(f"[Geometry] u0 = {self.detectorX_recon}, v0 = {self.detectorY_recon}")
        print(f"[Geometry] eta = {self.eta}, vc = {self.vc}, vs = {self.vs}")

        print("[Preprocess] Using Beer-Lambert projection: -log(I/I0)")
        print(f"[Preprocess] I0 = {self.I0}")
        print(f"[Preprocess] data shape = {self.data.shape}")
        print(f"[Preprocess] data min = {self.data.min():.6f}")
        print(f"[Preprocess] data max = {self.data.max():.6f}")
        print(f"[Preprocess] data mean = {self.data.mean():.6f}")

    def reconstruct(self):
        rec_gpu = None
        cuda_used = False
        try:
            cfg_fdk = ast.astra_dict("FDK_CUDA")
            cfg_fdk["ProjectionDataId"] = self.proj_id
            cfg_fdk["ReconstructionDataId"] = self.rec_id
            alg_id = ast.algorithm.create(cfg_fdk)
            ast.algorithm.run(alg_id, 1)
            rec_gpu = ast.data3d.get(self.rec_id)
            ar = rec_gpu[:]
            ast.algorithm.delete(alg_id)
            cuda_used = True
        except Exception as e:
            cuda_err = f"{type(e).__name__}: {str(e)}"
            print(
                f"[Warn] CUDA 不可用（{cuda_err[:80]}），降级为 CPU 重建（速度会变慢）"
            )
            try:
                cfg_cpu = ast.astra_dict("FDK")
                cfg_cpu["ProjectionDataId"] = self.proj_id
                cfg_cpu["ReconstructionDataId"] = self.rec_id
                alg_id = ast.algorithm.create(cfg_cpu)
                ast.algorithm.run(alg_id, 1)
                rec_cpu = ast.data3d.get(self.rec_id)
                ar = rec_cpu[:]
                ast.algorithm.delete(alg_id)
            except Exception as e2:
                cpu_err = f"{type(e2).__name__}: {str(e2)}"
                print(f"[Error] CPU 重建也失败：{cpu_err[:80]}")
                print(f"[Error] 请检查 ASTRA 是否正确安装，或确认有可用重建算法。")
                raise RuntimeError(
                    f"Both CUDA and CPU FDK reconstruction failed: {cpu_err}"
                )
        finally:
            ast.data3d.delete(self.rec_id)
            ast.data3d.delete(self.proj_id)

        if cuda_used:
            print("[Reconstruction] 使用 FDK_CUDA 重建完成")
        else:
            print("[Reconstruction] 使用 FDK (CPU) 重建完成")

        if self.use_hu:
            print("[HU] useHu=True was requested, but HU conversion is disabled")
        print("[HU] Disabled in current debug stage")
        print(f"[Reconstruction] min = {ar.min():.6f}")
        print(f"[Reconstruction] max = {ar.max():.6f}")
        print(f"[Reconstruction] mean = {ar.mean():.6f}")
        return ar


if __name__ == "__main__":
    import nibabel as nib
    import os

    output_dir = "/home/foods/pro/pyct_old/pyct/recon_output/phaseD_calibrated_v2/"
    os.makedirs(output_dir, exist_ok=True)

    cb = ConeBeam(
        SOD=908.8,
        TN=768,
        TM=972,
        SDD=959.6,
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
        eta=0.0,
        vc=-5.691,
        vs=-7.434,
        useHu=False,
        rescale_slope=1.0,
        rescale_intercept=0.0,
    )
    cb.load_img(angle_from_filename=True)
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

    nii_path_raw = os.path.join(output_dir, "phaseD_calibrated_v2_rec_raw.nii.gz")
    nii_img_raw = nib.Nifti1Image(rec_raw, np.eye(4))
    nib.save(nii_img_raw, nii_path_raw)
    print(f"Saved {nii_path_raw}")

    nii_path = os.path.join(output_dir, "phaseD_calibrated_v2_rec.nii.gz")
    nii_img = nib.Nifti1Image(rec_scaled, np.eye(4))
    nib.save(nii_img, nii_path)
    print(f"Saved {nii_path}")
