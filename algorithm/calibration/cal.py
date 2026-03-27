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


def detect_beads_log(img, n_expected=6, min_sigma=3.0, max_sigma=12.0, threshold=0.03):
    img_f = img.astype(np.float64)
    img_norm = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-12)
    blobs = blob_log(img_norm, min_sigma, max_sigma, num_sigma=10, threshold=threshold)
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
    circles = cv2.HoughCircles(
        blur,
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
    if not _HAS_ELLIPSE_MODEL:
        return np.ones(len(pts), dtype=bool)
    pts = np.array(pts)
    if len(pts) < 10:
        return np.ones(len(pts), dtype=bool)
    best_mask = np.zeros(len(pts), dtype=bool)
    best_count = 0
    for _ in range(n_iter):
        idx = np.random.choice(len(pts), 5, replace=False)
        sample = pts[idx]
        model = EllipseModel()
        if not model.estimate(sample):
            continue
        residuals = np.abs(model.residuals(pts))
        inliers = residuals < tol
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_mask = inliers
    if best_count < min_inliers:
        return np.ones(len(pts), dtype=bool)
    return best_mask


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
            blobs = detect_beads(img, self.num, **detect_kwargs)
            return fid, blobs

        with ThreadPoolExecutor(max_workers=50) as pool:
            results = list(pool.map(read_one, self.frame_ids))
        for fid, blobs in results:
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
        A = np.zeros((self.num, 2))
        for k in range(self.num):
            pts = clean_pts[k]
            if len(pts) < 5:
                continue
            ellipse = cv2.fitEllipse(pts.astype(np.float32))
            cx, cy = ellipse[0]
            a = max(ellipse[1]) / 2
            theta_e = ellipse[2] / 180 * pi
            A[k, 0] = cx + a * np.cos(theta_e)
            A[k, 1] = cy + a * np.sin(theta_e)
        tip_u = A[1:, 0]
        tip_v = A[1:, 1]
        b1, a1 = np.polyfit(tip_v, tip_u, 1)
        SDD = abs(b1) * self.dpixel
        v0 = -a1 / b1 if abs(b1) > 1e-9 else self.h / 2.0
        slope, intercept = np.polyfit(A[1:, 1], A[1:, 0], 1)
        theta = np.arctan(slope)
        u0 = intercept + slope * v0
        SOD_sum = 0.0
        count = 0
        for i in range(2, self.num):
            du = A[0, 0] - A[i, 0]
            dv = A[0, 1] - A[i, 1]
            dist = sqrt(du**2 + dv**2)
            if dist > 1e-3:
                SOD_sum += self.bead_spacing * (i - 1) * SDD / dist / self.dpixel
                count += 1
        SOD = SOD_sum / count if count > 0 else 0.0
        return SOD, SDD, u0, v0, theta

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

        x0 = np.array([SOD0, SDD0, u0_0, v0_0, theta0, eta0])
        w, h = self.w, self.h
        bounds = (
            [600, 700, w * 0.3, h * 0.3, -0.3, -0.1],
            [1500, 1800, w * 0.7, h * 0.7, 0.3, 0.1],
        )
        r_k_list = []
        phi0_k_list = []
        for k in range(self.num):
            pts = clean_pts[k]
            if len(pts) < 3:
                r_k_list.append(0.0)
                phi0_k_list.append(0.0)
                continue
            u_vals = pts[:, 0]
            r_est = (u_vals.max() - u_vals.min()) / 2.0 * self.dpixel * SOD0 / SDD0
            argmax_u = np.argmax(u_vals)
            phi0 = clean_angles[k][argmax_u]
            r_k_list.append(r_est)
            phi0_k_list.append(phi0)

        def residuals(params):
            SOD, SDD, u0, v0, theta, eta = params
            dpixel = self.dpixel
            res = []
            for k in range(self.num):
                pts = clean_pts[k]
                angles = clean_angles[k]
                if len(pts) < 3:
                    continue
                z_k = k * self.bead_spacing
                r_k = r_k_list[k]
                phi0_k = phi0_k_list[k]
                for ii, (phi, u_obs, v_obs) in enumerate(
                    zip(angles, pts[:, 0], pts[:, 1])
                ):
                    x_k = r_k * np.cos(phi + phi0_k)
                    y_k = r_k * np.sin(phi + phi0_k)
                    z_eff = z_k - eta * (SOD + y_k) * np.sin(phi)
                    denom = SOD + y_k
                    u_ideal = u0 + (SDD * x_k / denom) / dpixel
                    v_ideal = v0 + (SDD * z_eff / denom) / dpixel
                    du_rel = u_ideal - u0
                    dv_rel = v_ideal - v0
                    u_pred = u0 + np.cos(theta) * du_rel - np.sin(theta) * dv_rel
                    v_pred = v0 + np.sin(theta) * du_rel + np.cos(theta) * dv_rel
                    res.append(u_pred - u_obs)
                    res.append(v_pred - v_obs)
            return np.array(res)

        initial_res = residuals(x0)
        initial_rms = np.sqrt(np.mean(initial_res**2))
        print(f"Initial RMS: {initial_rms:.4f} pixels")
        result = least_squares(
            residuals,
            x0,
            bounds=bounds,
            method="trf",
            max_nfev=2000,
            ftol=1e-10,
            xtol=1e-10,
        )
        final_res = residuals(result.x)
        final_rms = np.sqrt(np.mean(final_res**2))
        print(f"Final RMS: {final_rms:.4f} pixels")
        names = ["SOD", "SDD", "u0", "v0", "theta", "eta"]
        for i, n in enumerate(names):
            print(f"  {n}: {result.x[i]:.6f} (change: {result.x[i] - x0[i]:+.6f})")
        return tuple(result.x)

    def calculate(self):
        if not self.detections:
            self.load_img()
        clean_pts, clean_angles = self._build_clean_trajectories()
        SOD0, SDD0, u0_0, v0_0, theta0 = self._legacy_estimate(clean_pts)
        eta0 = self._estimate_axis_tilt(clean_pts, SOD0, SDD0)
        SOD, SDD, u0, v0, theta, eta = self._refine(
            SOD0, SDD0, u0_0, v0_0, theta0, eta0, clean_pts, clean_angles
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
