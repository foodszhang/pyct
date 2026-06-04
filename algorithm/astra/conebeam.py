import cv2
import numpy as np
import astra as ast
import os
import pathlib
import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


def _ckpt(msg: str):
    """写入检查点日志"""
    log_path = pathlib.Path.home() / "pyct_crash.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {msg}\n")
    except OSError:
        pass


def _cuda_available() -> bool:
    """检测 ASTRA CUDA 是否可用"""
    try:
        info = ast.astra.get_gpu_info()
        return bool(info)
    except Exception:
        return False


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * max(float(mad), 1.0e-6)
    return np.abs(values - med) / scale


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
        rotation: float = 0.0,
        angle_offset_deg: float = 0.0,
        vol_center_x: float = 0.0,
        vol_center_y: float = 0.0,
        vol_center_z: float = 0.0,
        air_percentile: float | None = None,
        air_zero_percentile: float | None = None,
        mass_normalize: bool = False,
        projection_gaussian_sigma: float | None = None,
        projection_median_kernel: int = 0,
        intensity_floor: float = 1.0,
        intensity_clip_percentile: float | None = None,
        auto_defect_correction: bool = True,
        dark_filename: str = "dark.tif",
        empty_filename: str = "empty.tif",
        defect_pixel_z: float = 8.0,
        defect_line_z: float = 8.0,
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
        self.vol_center_x = vol_center_x
        self.vol_center_y = vol_center_y
        self.vol_center_z = vol_center_z
        xc = self.vol_center_x / self.voxel_size
        yc = self.vol_center_y / self.voxel_size
        zc = self.vol_center_z / self.voxel_size
        self.vol_geom = ast.create_vol_geom(
            NX,
            NY,
            NZ,
            -NY / 2.0 + xc,
            NY / 2.0 + xc,
            -NX / 2.0 + yc,
            NX / 2.0 + yc,
            -NZ / 2.0 + zc,
            NZ / 2.0 + zc,
        )
        self.rec_id = ast.data3d.create("-vol", self.vol_geom)
        self.ThreadPoolExecutor = ThreadPoolExecutor(max_workers=20)
        self.w = 0
        self.h = 0
        self.data_lock = Lock()
        self.use_hu = useHu
        self.rescale_slope = rescale_slope
        self.rescale_intercept = rescale_intercept
        self.I0 = 65535.0
        self.air_percentile = air_percentile
        self.air_zero_percentile = air_zero_percentile
        self.mass_normalize = mass_normalize
        self.projection_gaussian_sigma = projection_gaussian_sigma
        self.projection_median_kernel = projection_median_kernel
        self.intensity_floor = intensity_floor
        self.intensity_clip_percentile = intensity_clip_percentile
        self.projection_preprocess_stats = []
        self.auto_defect_correction = auto_defect_correction
        self.dark_filename = dark_filename
        self.empty_filename = empty_filename
        self.defect_pixel_z = defect_pixel_z
        self.defect_line_z = defect_line_z
        self.defect_map = None
        if self.auto_defect_correction:
            self.defect_map = self._build_auto_defect_map()
        self.eta = eta
        self.vc = vc
        self.vs = vs
        self.rotation = rotation
        self.angle_offset_deg = angle_offset_deg
        print(f"[Geometry] eta = {self.eta}, vc = {self.vc}, vs = {self.vs}")
        print(f"[Geometry] detector_roll_deg = {self.rotation}")
        print(f"[Geometry] angle_offset_deg = {self.angle_offset_deg}")
        print(
            f"[Geometry] vol_center = ({self.vol_center_x}, {self.vol_center_y}, {self.vol_center_z}) mm"
        )

    def _build_auto_defect_map(self) -> dict | None:
        dark_path = os.path.join(self.proj_path, self.dark_filename)
        empty_path = os.path.join(self.proj_path, self.empty_filename)
        dark = cv2.imread(dark_path, -1)
        empty = cv2.imread(empty_path, -1)
        if dark is None or empty is None:
            print(
                f"[Warn] 自动坏点坏线校正已启用，但未找到 {self.dark_filename}/{self.empty_filename}"
            )
            return None
        dark = dark.astype(np.float32)
        empty = empty.astype(np.float32)
        if dark.shape != empty.shape:
            print(f"[Warn] dark/empty 尺寸不一致，跳过自动坏点坏线校正: {dark.shape} vs {empty.shape}")
            return None

        response = empty - dark
        response = np.where(np.isfinite(response), response, 0.0)
        valid = response > max(float(np.percentile(response, 5.0)), 1.0)
        local_response = cv2.medianBlur(response, 5)
        rel_dev = np.abs(response - local_response) / np.maximum(local_response, 1.0)

        dark_z = _robust_z(dark)
        response_med = float(np.median(response[valid])) if np.any(valid) else float(np.median(response))
        response_mad = float(np.median(np.abs(response[valid] - response_med))) if np.any(valid) else 1.0
        response_scale = 1.4826 * max(response_mad, 1.0e-6)
        response_z_full = np.abs(response - response_med) / response_scale

        bad_pixels = (
            (dark_z > self.defect_pixel_z)
            | (response_z_full > self.defect_pixel_z)
            | (response <= max(1.0, response_med * 0.05))
            | ((rel_dev > 0.35) & (response_z_full > 4.0))
        )

        row_metric = np.median(response, axis=1)
        col_metric = np.median(response, axis=0)
        bad_rows = _robust_z(row_metric) > self.defect_line_z
        bad_cols = _robust_z(col_metric) > self.defect_line_z
        bad_rows |= row_metric <= max(1.0, response_med * 0.05)
        bad_cols |= col_metric <= max(1.0, response_med * 0.05)

        # If an entire bad line was found, do not double-count every pixel on it as isolated bad pixels.
        bad_pixels[bad_rows, :] = False
        bad_pixels[:, bad_cols] = False

        defect_map = {
            "shape": dark.shape,
            "bad_pixels": bad_pixels,
            "bad_rows": bad_rows,
            "bad_cols": bad_cols,
            "response_median": response_med,
            "bad_pixel_count": int(np.count_nonzero(bad_pixels)),
            "bad_row_count": int(np.count_nonzero(bad_rows)),
            "bad_col_count": int(np.count_nonzero(bad_cols)),
        }
        print(
            "[Defect] 自动坏点坏线检测完成: "
            f"bad_pixels={defect_map['bad_pixel_count']}, "
            f"bad_rows={defect_map['bad_row_count']}, "
            f"bad_cols={defect_map['bad_col_count']}"
        )
        return defect_map

    def _interpolate_bad_lines(self, img: np.ndarray, bad_rows: np.ndarray, bad_cols: np.ndarray) -> np.ndarray:
        out = img.copy()
        good_rows = np.flatnonzero(~bad_rows)
        for row in np.flatnonzero(bad_rows):
            if good_rows.size == 0:
                break
            left = good_rows[good_rows < row]
            right = good_rows[good_rows > row]
            if left.size and right.size:
                out[row, :] = 0.5 * (out[left[-1], :] + out[right[0], :])
            elif left.size:
                out[row, :] = out[left[-1], :]
            elif right.size:
                out[row, :] = out[right[0], :]

        good_cols = np.flatnonzero(~bad_cols)
        for col in np.flatnonzero(bad_cols):
            if good_cols.size == 0:
                break
            left = good_cols[good_cols < col]
            right = good_cols[good_cols > col]
            if left.size and right.size:
                out[:, col] = 0.5 * (out[:, left[-1]] + out[:, right[0]])
            elif left.size:
                out[:, col] = out[:, left[-1]]
            elif right.size:
                out[:, col] = out[:, right[0]]
        return out

    def _repair_defects(self, img: np.ndarray) -> np.ndarray:
        if not self.defect_map or img.shape != self.defect_map["shape"]:
            return img
        repaired = self._interpolate_bad_lines(
            img,
            self.defect_map["bad_rows"],
            self.defect_map["bad_cols"],
        )
        bad_pixels = self.defect_map["bad_pixels"]
        if np.any(bad_pixels):
            median_img = cv2.medianBlur(repaired, 3)
            repaired[bad_pixels] = median_img[bad_pixels]
        return repaired

    def _preprocess_projection(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        simg = img.astype(np.float32)
        if self.auto_defect_correction:
            simg = self._repair_defects(simg)
        TM, TN = simg.shape
        self.w = TN
        self.h = TM
        interp = cv2.INTER_AREA if (self.TN < TN or self.TM < TM) else cv2.INTER_LINEAR
        reshaped = cv2.resize(simg, (self.TN, self.TM), interpolation=interp)

        stats = {
            "raw_min": float(np.min(simg)),
            "raw_max": float(np.max(simg)),
            "raw_zero_fraction": float(np.mean(simg <= 0.0)),
            "raw_saturated_fraction": float(np.mean(simg >= self.I0)),
            "defect_correction": bool(self.auto_defect_correction and self.defect_map is not None),
        }

        if self.projection_gaussian_sigma is not None and self.projection_gaussian_sigma > 0.0:
            sigma = float(self.projection_gaussian_sigma)
            reshaped = cv2.GaussianBlur(
                reshaped,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )

        if self.projection_median_kernel and self.projection_median_kernel > 1:
            kernel = int(self.projection_median_kernel)
            if kernel % 2 == 0:
                kernel += 1
            reshaped = cv2.medianBlur(reshaped, kernel)

        clip_hi = self.I0
        if self.intensity_clip_percentile is not None:
            clip_hi = float(np.percentile(reshaped, self.intensity_clip_percentile))
            clip_hi = min(max(clip_hi, 1.0), self.I0)
        floor = max(float(self.intensity_floor), 1.0)
        reshaped = np.clip(reshaped, floor, clip_hi)

        I0 = self.I0
        if self.air_percentile is not None:
            I0 = float(np.percentile(reshaped, self.air_percentile))
            I0 = max(I0, 1.0)
        projection = -np.log(np.clip(reshaped / I0, 1e-6, 1.0))
        stats.update(
            {
                "clip_hi": float(clip_hi),
                "floor": float(floor),
                "I0": float(I0),
                "log_min": float(np.min(projection)),
                "log_max": float(np.max(projection)),
                "log_mean": float(np.mean(projection)),
            }
        )
        return projection, stats

    def projection_preprocess_summary(self) -> dict:
        if not self.projection_preprocess_stats:
            return {}
        keys = [
            "raw_zero_fraction",
            "raw_saturated_fraction",
            "clip_hi",
            "I0",
            "log_min",
            "log_max",
            "log_mean",
        ]
        summary = {"count": len(self.projection_preprocess_stats)}
        for key in keys:
            values = np.array(
                [row[key] for row in self.projection_preprocess_stats if key in row],
                dtype=np.float32,
            )
            if values.size == 0:
                continue
            summary[key] = {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        if self.defect_map is not None:
            summary["defect_map"] = {
                "bad_pixels": self.defect_map["bad_pixel_count"],
                "bad_rows": self.defect_map["bad_row_count"],
                "bad_cols": self.defect_map["bad_col_count"],
                "response_median": self.defect_map["response_median"],
            }
        return summary

    def load_from_dict(self, img_dict):
        self.projection_preprocess_stats = []
        self.data = np.zeros((self.TM, len(img_dict), self.TN), dtype=np.float32)
        img_list = list(img_dict.items())
        img_list = sorted(img_list, key=lambda x: x[0])

        self.detectorX_recon = self.detectorX_raw * self.sx
        self.detectorY_recon = self.detectorY_raw * self.sy
        self.dd_x_recon = self.pixel_size_raw / self.sx
        self.dd_y_recon = self.pixel_size_raw / self.sy

        for n, v in enumerate(img_list):
            i, img = v
            projection, stats = self._preprocess_projection(img)
            stats["angle"] = float(i)
            self.projection_preprocess_stats.append(stats)
            self.data[:, n, :] = projection
        angles = [item[0] for item in img_list]
        angles = [(float(i) + self.angle_offset_deg) * np.pi / 180.0 for i in angles]

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
            rotation=self.rotation,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)

    def build_cone_vec(
        self, angles, SOD, SDD, u0, v0, eta=0.0, vc=0.0, vs=0.0, rotation=0.0
    ):
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
        - rotation: detector roll angle in degrees. Positive roll rotates u toward v.
        """
        ODD = SDD - SOD

        du = self.pixel_size_raw / self.sx
        dv = self.pixel_size_raw / self.sy
        gamma = np.deg2rad(rotation)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        n_angles = len(angles)
        vectors = np.zeros((n_angles, 12))

        for i, phi in enumerate(angles):
            sp = np.sin(phi)
            cp = np.cos(phi)

            srcX = sp * SOD / self.voxel_size
            srcY = -cp * SOD / self.voxel_size
            srcZ = 0.0

            u_base = np.array([cp * du, sp * du, 0.0]) / self.voxel_size
            v_base = np.array([-eta * sp * dv, eta * cp * dv, -dv]) / self.voxel_size
            u_vec = cos_g * u_base + sin_g * v_base
            v_vec = -sin_g * u_base + cos_g * v_base
            uX, uY, uZ = u_vec
            vX, vY, vZ = v_vec

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
            projection, stats = self._preprocess_projection(img)
            stats["filename"] = filename
            self.data_lock.acquire()
            self.data[:, number, :] = projection
            self.projection_preprocess_stats.append(stats)
            self.data_lock.release()
        else:
            print(f"{full_path} not exists")

    def load_img(
        self,
        angle_from_filename: bool = False,
        progress_callback=None,
        drop_duplicate_360: bool = False,
        fill_missing_degrees: bool = False,
    ):
        self.projection_preprocess_stats = []
        if angle_from_filename:
            tif_files = [f for f in os.listdir(self.proj_path) if f.endswith(".tif")]
            parsed = []
            for f in tif_files:
                try:
                    angle_deg = float(f.replace(".tif", ""))
                    if drop_duplicate_360 and abs(angle_deg - 360.0) < 1e-6:
                        continue
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
            angles = [(a + self.angle_offset_deg) * np.pi / 180.0 for a in angle_deg_list]
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
            angles = [
                (i * 360.0 / self.number_of_img + self.angle_offset_deg)
                * np.pi
                / 180.0
                for i in range(count)
            ]
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

        if angle_from_filename and fill_missing_degrees:
            target_degrees = np.arange(0, 360, dtype=np.float32)
            src_degrees = np.asarray(angle_deg_list, dtype=np.float32)
            order = np.argsort(src_degrees)
            src_degrees = src_degrees[order]
            src_data = self.data[:, order, :]
            filled = np.empty((self.TM, len(target_degrees), self.TN), dtype=np.float32)
            src_index = {int(round(a)): i for i, a in enumerate(src_degrees)}
            for out_i, degree in enumerate(target_degrees):
                degree_i = int(degree)
                if degree_i in src_index:
                    filled[:, out_i, :] = src_data[:, src_index[degree_i], :]
                    continue
                right = int(np.searchsorted(src_degrees, degree, side="right"))
                if right <= 0:
                    left = len(src_degrees) - 1
                    right = 0
                    span = (src_degrees[right] + 360.0) - src_degrees[left]
                    weight = (degree + 360.0 - src_degrees[left]) / span
                elif right >= len(src_degrees):
                    left = len(src_degrees) - 1
                    right = 0
                    span = (src_degrees[right] + 360.0) - src_degrees[left]
                    weight = (degree - src_degrees[left]) / span
                else:
                    left = right - 1
                    span = src_degrees[right] - src_degrees[left]
                    weight = (degree - src_degrees[left]) / span
                filled[:, out_i, :] = (
                    (1.0 - weight) * src_data[:, left, :]
                    + weight * src_data[:, right, :]
                )
            self.data = filled
            angles = [(a + self.angle_offset_deg) * np.pi / 180.0 for a in target_degrees]
            count = len(target_degrees)
            print(
                f"[Angles] 线性补齐缺失角度: {len(src_degrees)} -> {count} 张"
            )

        if self.air_zero_percentile is not None:
            for i in range(self.data.shape[1]):
                base = float(np.percentile(self.data[:, i, :], self.air_zero_percentile))
                self.data[:, i, :] = np.clip(self.data[:, i, :] - base, 0.0, None)
            print(f"[Preprocess] 每角度空气基线归零 p={self.air_zero_percentile}")

        if self.mass_normalize:
            metrics = np.zeros(self.data.shape[1], dtype=np.float32)
            for i in range(self.data.shape[1]):
                img = self.data[:, i, :]
                metrics[i] = float(np.mean(np.clip(img, 0.0, np.percentile(img, 99.5))))
            target = float(np.median(metrics[metrics > 1e-8]))
            for i, metric in enumerate(metrics):
                if metric <= 1e-8:
                    continue
                scale = np.clip(target / float(metric), 0.75, 1.33)
                self.data[:, i, :] *= scale
            print("[Preprocess] 每角度总衰减积分归一化完成")
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
            rotation=self.rotation,
        )
        self.proj_geom = ast.create_proj_geom("cone_vec", self.TM, self.TN, vectors)
        self.proj_id = ast.data3d.create("-proj3d", self.proj_geom, self.data)

        print("[Geometry] Using cone_vec geometry (no postalignment)")
        print(f"[Geometry] SOD = {self.SOD}, SDD = {self.SDD}")
        print(f"[Geometry] u0 = {self.detectorX_recon}, v0 = {self.detectorY_recon}")
        print(f"[Geometry] eta = {self.eta}, vc = {self.vc}, vs = {self.vs}")
        print(f"[Geometry] detector_roll_deg = {self.rotation}")
        print(f"[Geometry] angle_offset_deg = {self.angle_offset_deg}")

        print("[Preprocess] Using Beer-Lambert projection: -log(I/I0)")
        print(f"[Preprocess] I0 = {self.I0}")
        if self.air_percentile is not None:
            print(f"[Preprocess] air_percentile = {self.air_percentile}")
        if self.projection_gaussian_sigma is not None:
            print(f"[Preprocess] gaussian_sigma = {self.projection_gaussian_sigma}")
        if self.projection_median_kernel:
            print(f"[Preprocess] median_kernel = {self.projection_median_kernel}")
        if self.intensity_clip_percentile is not None:
            print(f"[Preprocess] intensity_clip_percentile = {self.intensity_clip_percentile}")
        print(f"[Preprocess] intensity_floor = {self.intensity_floor}")
        print(f"[Preprocess] data shape = {self.data.shape}")
        print(f"[Preprocess] data min = {self.data.min():.6f}")
        print(f"[Preprocess] data max = {self.data.max():.6f}")
        print(f"[Preprocess] data mean = {self.data.mean():.6f}")

    def reconstruct(self, *, filter_type="Ram-Lak", algorithm="FDK",
                    iterations=50, non_neg_constraint=True,
                    ring_correction=False, ring_kernel_size=9):
        use_cuda = _cuda_available()
        _ckpt(f"CUDA available: {use_cuda}")

        if not use_cuda:
            raise RuntimeError(
                "GPU不可用：astra.use_cuda()返回False。\n"
                "请检查：\n"
                "1. 显卡驱动已正确安装（RTX 2080 Ti 需要 ≥ 465.89）\n"
                "2. CUDA toolkit 与驱动版本匹配\n"
                "3. astra.test() 输出的报错信息"
            )

        # 环形伪影校正
        if ring_correction:
            from scipy.ndimage import median_filter
            for row in range(self.data.shape[0]):
                sino_row = self.data[row, :, :]
                row_mean = sino_row.mean(axis=0)
                row_mean_filtered = median_filter(row_mean, size=ring_kernel_size)
                correction = row_mean - row_mean_filtered
                self.data[row, :, :] -= correction[np.newaxis, :]
            ast.data3d.store(self.proj_id, self.data)
            print(f"[Preprocess] 环形伪影校正完成，核大小={ring_kernel_size}")

        filter_map = {
            "Ram-Lak": "ram-lak",
            "Shepp-Logan": "shepp-logan",
            "Cosine": "cosine",
            "Hamming": "hamming",
            "Hann": "hann",
        }

        _ckpt("FDK_CUDA path start")
        try:
            if algorithm == "FDK":
                cfg_fdk = ast.astra_dict("FDK_CUDA")
                cfg_fdk["ProjectionDataId"] = self.proj_id
                cfg_fdk["ReconstructionDataId"] = self.rec_id
                cfg_fdk["option"] = {}
                cfg_fdk["option"]["FilterType"] = filter_map.get(filter_type, "ram-lak")
                _ckpt("FDK_CUDA algorithm create start")
                alg_id = ast.algorithm.create(cfg_fdk)
                _ckpt("FDK_CUDA algorithm run start")
                ast.algorithm.run(alg_id, 1)
                _ckpt("FDK_CUDA get result start")
                rec_gpu = ast.data3d.get(self.rec_id)
                ar = rec_gpu[:]
                ast.algorithm.delete(alg_id)
                _ckpt("FDK_CUDA done")
                print(f"[Reconstruction] 使用 FDK_CUDA 滤波器={filter_type} 重建完成")
            elif algorithm == "SIRT":
                cfg = ast.astra_dict("SIRT3D_CUDA")
                cfg["ProjectionDataId"] = self.proj_id
                cfg["ReconstructionDataId"] = self.rec_id
                cfg["option"] = {}
                if non_neg_constraint:
                    cfg["option"]["MinConstraint"] = 0.0
                alg_id = ast.algorithm.create(cfg)
                ast.algorithm.run(alg_id, iterations)
                rec_gpu = ast.data3d.get(self.rec_id)
                ar = rec_gpu[:]
                ast.algorithm.delete(alg_id)
                print(f"[Reconstruction] 使用 SIRT3D_CUDA {iterations} 次迭代完成")
            elif algorithm == "CGLS":
                cfg = ast.astra_dict("CGLS3D_CUDA")
                cfg["ProjectionDataId"] = self.proj_id
                cfg["ReconstructionDataId"] = self.rec_id
                alg_id = ast.algorithm.create(cfg)
                ast.algorithm.run(alg_id, iterations)
                rec_gpu = ast.data3d.get(self.rec_id)
                ar = rec_gpu[:]
                ast.algorithm.delete(alg_id)
                print(f"[Reconstruction] 使用 CGLS3D_CUDA {iterations} 次迭代完成")
        finally:
            ast.data3d.delete(self.rec_id)
            ast.data3d.delete(self.proj_id)

        # 非负约束后处理（FDK 路径）
        if algorithm == "FDK" and non_neg_constraint:
            ar = np.clip(ar, 0, None)
            print("[Postprocess] 非负约束：裁剪负值")

        if self.use_hu:
            print("[HU] useHu=True was requested, but HU conversion is disabled")
            print("[HU] Disabled in current debug stage")
        print(f"[Reconstruction] min = {ar.min():.6f}")
        print(f"[Reconstruction] max = {ar.max():.6f}")
        print(f"[Reconstruction] mean = {ar.mean():.6f}")
        return ar, use_cuda


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
        vol_center_x=0.0,
        vol_center_y=0.0,
        vol_center_z=0.0,
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
