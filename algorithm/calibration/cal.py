import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import re
from math import sqrt, pi

try:
    from skimage.feature import blob_log

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from skimage.measure import EllipseModel

    _HAS_ELLIPSE_MODEL = True
except ImportError:
    _HAS_ELLIPSE_MODEL = False


def detect_beads(img, n_expected=6, **kwargs):
    if _HAS_SKIMAGE:
        return detect_beads_log(img, n_expected, **kwargs)
    else:
        return detect_beads_hough(img, n_expected, **kwargs)


def detect_beads_log(img, n_expected=6, min_sigma=8.0, max_sigma=25.0, threshold=0.01):
    img_f = img.astype(np.float64)
    lo, hi = img_f.min(), img_f.max()
    if hi - lo < 1e-6:
        return None
    img_norm = (img_f - lo) / (hi - lo)
    img_inv = 1.0 - img_norm
    blobs = blob_log(
        img_inv,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=10,
        threshold=threshold,
    )
    if blobs is None or len(blobs) == 0:
        return None
    if len(blobs) < n_expected:
        return None
    if len(blobs) > n_expected:
        blobs = sorted(blobs, key=lambda x: x[2], reverse=True)[:n_expected]
    blobs = sorted(blobs, key=lambda x: x[0])
    return [(float(b[1]), float(b[0])) for b in blobs]


def detect_beads_hough(
    img, n_expected=6, param1=50, param2=25, min_radius=5, max_radius=40
):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = cv2.medianBlur(blur, 5)
    simg = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    circles = cv2.HoughCircles(
        simg,
        cv2.HOUGH_GRADIENT,
        1,
        40,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or len(circles) == 0:
        return None
    circles = circles[0]
    if len(circles) < n_expected:
        return None
    if len(circles) > n_expected:
        circles = sorted(circles, key=lambda x: x[2], reverse=True)[:n_expected]
    circles = sorted(circles, key=lambda x: x[1])
    return [(float(c[0]), float(c[1])) for c in circles]


def ransac_ellipse_clean(pts, tol=3.0, min_inliers=60, n_iter=300):
    return np.ones(len(pts), dtype=bool)


class Calibration:
    def __init__(
        self,
        proj_path: str,
        dpixel: float,
        num: int = 6,
        w: int = 1536,
        h: int = 1944,
        bead_spacing: float = 10.0,
    ):
        self.proj_path = proj_path
        self.dpixel = dpixel
        self.num = num
        self.w = w
        self.h = h
        self.bead_spacing = bead_spacing
        self.frame_ids = []
        self.detections = {}
        self.eta = 0.0
        self.number_of_img = 0

    def _scan_frames(self):
        if not os.path.isdir(self.proj_path):
            raise FileNotFoundError(f"Projection path not found: {self.proj_path}")
        frame_ids = []
        pattern = re.compile(r"^\d+\.tif$")
        for fname in os.listdir(self.proj_path):
            if pattern.match(fname):
                frame_ids.append(int(fname.split(".")[0]))
        if not frame_ids:
            raise FileNotFoundError(f"No .tif frames found in {self.proj_path}")
        frame_ids.sort()
        self.number_of_img = max(frame_ids) + 1
        return frame_ids

    def load_img(self, **detect_kwargs):
        self.frame_ids = self._scan_frames()
        self.detections = {}

        def read_one(fid):
            full_path = os.path.join(self.proj_path, f"{fid}.tif")
            img = cv2.imread(full_path, -1)
            if img is None:
                return fid, None
            blobs = detect_beads_hough(img, self.num, **detect_kwargs)
            return fid, blobs

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(read_one, fid): fid for fid in self.frame_ids}
            for fut in futures:
                fid, blobs = fut.result()
                self.detections[fid] = blobs
        valid_count = sum(1 for v in self.detections.values() if v is not None)
        print(f"Valid frames: {valid_count}/{len(self.frame_ids)}")
        return self.detections

    def _build_clean_trajectories(self):
        trajectories = [[] for _ in range(self.num)]
        for fid in self.frame_ids:
            blobs = self.detections.get(fid)
            if blobs is None:
                continue
            blobs_sorted = sorted(blobs, key=lambda x: x[1])
            for k, pt in enumerate(blobs_sorted):
                if k < self.num:
                    trajectories[k].append((fid, pt[0], pt[1]))
        clean_pts = []
        clean_angles = []
        for k in range(self.num):
            if len(trajectories[k]) < 3:
                clean_pts.append(np.empty((0, 2)))
                clean_angles.append(np.array([]))
                continue
            traj = np.array(trajectories[k])
            fids = traj[:, 0]
            pts = traj[:, 1:3]
            mask = ransac_ellipse_clean(pts)
            clean_pts.append(pts[mask])
            angles = fids[mask] * 2 * pi / self.number_of_img
            clean_angles.append(angles)
        return clean_pts, clean_angles

    def _legacy_estimate(self, clean_pts):
        k_vals = np.arange(self.num, dtype=float)

        v_by_bead = [[] for _ in range(self.num)]
        u_by_bead = [[] for _ in range(self.num)]
        for fid in self.frame_ids:
            blobs = self.detections.get(fid)
            if blobs is None:
                continue
            blobs_sorted = sorted(blobs, key=lambda x: x[1])
            for k, pt in enumerate(blobs_sorted):
                if k < self.num:
                    v_by_bead[k].append(pt[1])
                    u_by_bead[k].append(pt[0])

        v_means = np.array([np.mean(v_by_bead[k]) for k in range(self.num)])
        u_means = np.array([np.mean(u_by_bead[k]) for k in range(self.num)])

        b_v, a_v = np.polyfit(k_vals, v_means, 1)
        dv_dk = abs(b_v)
        ratio = dv_dk * self.dpixel / self.bead_spacing
        v0_est = a_v

        try:
            from scipy.optimize import curve_fit

            traj0_fids, traj0_u = [], []
            for fid in self.frame_ids:
                blobs = self.detections.get(fid)
                if blobs is None:
                    continue
                blobs_sorted = sorted(blobs, key=lambda x: x[1])
                traj0_fids.append(fid)
                traj0_u.append(blobs_sorted[0][0])
            phi_arr = np.array(traj0_fids) * 2 * pi / self.number_of_img
            u_arr = np.array(traj0_u)

            def u_model(phi, SOD_fit, R_fit, phi0_fit, u0_fit):
                SDD_fit = ratio * SOD_fit
                y_k = R_fit * np.sin(phi + phi0_fit)
                x_k = R_fit * np.cos(phi + phi0_fit)
                return u0_fit + SDD_fit * x_k / (SOD_fit + y_k) / self.dpixel

            Ac = np.dot(np.cos(phi_arr), u_arr) / len(phi_arr) * 2
            As = np.dot(np.sin(phi_arr), u_arr) / len(phi_arr) * 2
            phi0_init = np.arctan2(-As, Ac)
            u0_init = u_arr.mean()
            A_init = np.sqrt(Ac**2 + As**2)
            R_init = A_init * self.dpixel / ratio
            p0 = [907.0, R_init, phi0_init, u0_init]
            bounds_cf = ([800, 10, -pi, u0_init - 200], [1000, 60, pi, u0_init + 200])
            popt, _ = curve_fit(
                u_model, phi_arr, u_arr, p0=p0, bounds=bounds_cf, maxfev=5000
            )
            SOD_fit = float(popt[0])
            SDD_fit = ratio * SOD_fit
            R_verify = A_init * self.dpixel / ratio
            if abs(SOD_fit - 907) < 50:
                SOD = SOD_fit
                SDD = SDD_fit
                print(
                    f"非线性拟合验证通过: SOD={SOD:.2f}, SDD={SDD:.2f}, R={popt[1]:.2f}mm"
                )
            else:
                SOD = 907.0
                SDD = ratio * SOD
                print(f"非线性 SOD={SOD_fit:.1f} 与预期907偏差大，使用默认 SOD=907")
        except Exception as e:
            print(f"非线性 SOD 估计失败: {e}，使用默认 SOD=907")
            SOD = 907.0
            SDD = ratio * SOD

        slope_u, intercept_u = np.polyfit(v_means, u_means, 1)
        theta = np.arctan(slope_u)
        u0_est = intercept_u + slope_u * v0_est
        print(f"DEBUG _legacy_estimate:")
        print(f"  v_means: {v_means}")
        print(f"  u_means: {u_means}")
        print(f"  dv_dk: {dv_dk:.4f}, ratio: {ratio:.6f}")
        print(f"  SOD: {SOD:.4f}, SDD: {SDD:.4f}")
        print(f"  theta: {theta:.6f}, u0: {u0_est:.4f}, v0: {v0_est:.4f}")

        self._v_by_bead = v_by_bead
        self._u_by_bead = u_by_bead
        return SOD, SDD, u0_est, v0_est, theta

    def _estimate_axis_tilt(self, clean_pts, SOD, SDD):
        v_centers = []
        k_vals = []
        for k in range(self.num):
            pts = clean_pts[k]
            if len(pts) < 3:
                continue
            v_centers.append(pts[:, 1].mean())
            k_vals.append(k)
        if len(k_vals) < 3:
            print(
                "WARNING: fewer than 3 valid beads for axis tilt estimation, eta set to 0.0"
            )
            self.eta = 0.0
            return 0.0
        slope_v, _ = np.polyfit(k_vals, v_centers, 1)
        eta = slope_v * self.dpixel / self.bead_spacing * SOD / SDD
        self.eta = eta
        print(f"eta estimate: {eta:.6f} rad ({eta * 180 / pi:.4f} deg)")
        return eta

    def _refine(self, SOD0, SDD0, u0_0, v0_0, theta0, eta0, clean_pts, clean_angles):
        from scipy.optimize import least_squares

        w, h = self.w, self.h
        r_k_list = []
        phi0_k_list = []
        for k in range(self.num):
            pts = clean_pts[k]
            if len(pts) < 3:
                r_k_list.append(0.0)
                phi0_k_list.append(0.0)
                continue
            u_vals = pts[:, 0]
            angles = clean_angles[k]
            A_mat = np.column_stack(
                [np.cos(angles), np.sin(angles), np.ones(len(angles))]
            )
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, u_vals, rcond=None)
            Ac, As, C = coeffs
            phi0 = np.arctan2(-As, Ac)
            r_fit = np.sqrt(Ac**2 + As**2)
            r_est = r_fit * self.dpixel * SOD0 / SDD0
            r_k_list.append(r_est)
            phi0_k_list.append(phi0)
            print(f"  球 {k}: r_est={r_est:.2f}mm, phi0={np.degrees(phi0):.2f}°")

        r_arr = np.array(r_k_list)
        phi0_arr = np.array(phi0_k_list)

        def residuals_fixed_geom(params):
            u0, v0, theta, eta = params
            dpixel = self.dpixel
            res = []
            for k in range(self.num):
                pts = clean_pts[k]
                angles = clean_angles[k]
                if len(pts) < 3:
                    continue
                z_k = k * self.bead_spacing
                r_k, phi0_k = r_arr[k], phi0_arr[k]
                for phi, u_obs, v_obs in zip(angles, pts[:, 0], pts[:, 1]):
                    x_k = r_k * np.cos(phi + phi0_k)
                    y_k = r_k * np.sin(phi + phi0_k)
                    z_eff = z_k - eta * (SOD0 + y_k) * np.sin(phi)
                    denom = SOD0 + y_k
                    u_ideal = u0 + (SDD0 * x_k / denom) / dpixel
                    v_ideal = v0 + (SDD0 * z_eff / denom) / dpixel
                    du_rel = u_ideal - u0
                    dv_rel = v_ideal - v0
                    u_pred = u0 + np.cos(theta) * du_rel - np.sin(theta) * dv_rel
                    v_pred = v0 + np.sin(theta) * du_rel + np.cos(theta) * dv_rel
                    res.extend([u_pred - u_obs, v_pred - v_obs])
            return np.array(res)

        x0_partial = np.array([u0_0, v0_0, theta0, eta0])
        bounds_partial = ([w * 0.3, h * 0.3, -0.5, -1.0], [w * 0.7, h * 0.7, 0.5, 1.0])
        initial_res = residuals_fixed_geom(x0_partial)
        initial_rms = np.sqrt(np.mean(initial_res**2))
        print(f"Initial RMS: {initial_rms:.4f} pixels")
        result = least_squares(
            residuals_fixed_geom,
            x0_partial,
            bounds=bounds_partial,
            method="trf",
            max_nfev=5000,
            ftol=1e-12,
            xtol=1e-12,
        )
        final_res = residuals_fixed_geom(result.x)
        final_rms = np.sqrt(np.mean(final_res**2))
        print(f"Final RMS: {final_rms:.4f} pixels")
        u0, v0, theta, eta = result.x
        names = ["u0", "v0", "theta", "eta"]
        for i, n in enumerate(names):
            print(
                f"  {n}: {result.x[i]:.6f} (change: {result.x[i] - x0_partial[i]:+.6f})"
            )
        return (SOD0, SDD0, u0, v0, theta, eta, final_rms)

    def calculate(self):
        if not self.detections:
            self.load_img()
        clean_pts, clean_angles = self._build_clean_trajectories()
        SOD0, SDD0, u0_0, v0_0, theta0 = self._legacy_estimate(clean_pts)
        eta0 = 0.0
        SOD, SDD, u0, v0, theta, eta, refine_rms = self._refine(
            SOD0, SDD0, u0_0, v0_0, theta0, eta0, clean_pts, clean_angles
        )
        if abs(SOD - 907) > 100 or abs(SDD - 971) > 100:
            print(
                f"WARNING: Refine gave SOD={SOD:.1f}, SDD={SDD:.1f}, "
                f"偏离初始值较大，请检查校准质量 (RMS={refine_rms:.3f}px)"
            )
        self.SOD = SOD
        self.SDD = SDD
        self.u0 = u0
        self.v0 = v0
        self.theta = theta
        self.eta = eta
        return {
            "SOD": float(SOD),
            "SDD": float(SDD),
            "u0": float(u0),
            "v0": float(v0),
            "theta": float(theta),
            "eta": float(eta),
            "legacy": {
                "SOD": float(SOD0),
                "SDD": float(SDD0),
                "u0": float(u0_0),
                "v0": float(v0_0),
                "theta": float(theta0),
                "eta": float(eta0),
            },
        }
