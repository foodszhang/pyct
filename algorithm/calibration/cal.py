# algorithm/calibration/cal.py

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import numba as nb
from scipy.optimize import minimize_scalar, least_squares


@nb.jit(nopython=True)
def circle_threshold(img, circle):
    max_val = 0
    min_val = 255
    for i in range(int(circle[1] - circle[2]) - 2, int(circle[1] + circle[2]) + 2):
        for j in range(int(circle[0] - circle[2]) - 1, int(circle[0] + circle[2]) + 2):
            distance = np.sqrt((i - circle[1]) ** 2 + (j - circle[0]) ** 2)
            if distance <= circle[2]:
                if img[i, j] > max_val:
                    max_val = img[i, j]
                if img[i, j] < min_val:
                    min_val = img[i, j]
    return min_val, max_val


def hough_circles(img):
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=20, minRadius=0, maxRadius=0
    )
    if circles is None:
        return []
    return circles[0]


class Calibration:
    def __init__(self, proj_path, dpixel, num, w, h):
        self.proj_path = proj_path
        self.dpixel = dpixel
        self.num = num
        self.w = w
        self.h = h
        self.zero_img = np.zeros((h, w, 3), dtype=np.uint8)
        self._observations = None

    def read_circle(self, i):
        """读取第 i 张投影，返回 6 个珠子的 (u, v) 像素坐标列表"""
        if not os.path.exists(os.path.join(self.proj_path, f"{i}.tif")):
            return None
        img = cv2.imread(os.path.join(self.proj_path, f"{i}.tif"), -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 5)
        simg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        simg = simg.astype(np.uint8)
        circles = hough_circles(simg)
        if len(circles) != 6 and len(circles) != 0:
            threshold = circle_threshold(simg, circles[0])
            th = np.where((simg <= threshold[1]) & (simg >= threshold[0]), simg, 255)
            circles = hough_circles(th)
            if len(circles) != 6:
                return None
        return [(x[0], x[1]) for x in circles]

    def load_img(self):
        """加载所有投影，返回观测列表: [(phi, bead_idx, u, v), ...]"""
        if self._observations is not None:
            return self._observations

        pool = ThreadPoolExecutor(max_workers=50)
        fut_list = []

        for i in range(0, 360):
            fut = pool.submit(self.read_circle, i)
            fut_list.append((i, fut))

        observations = []
        for angle_idx, fut in fut_list:
            points = fut.result()
            if points is None:
                continue
            points = sorted(points, key=lambda x: x[1])
            phi = angle_idx * 2 * np.pi / 360
            for bead_idx, (u, v) in enumerate(points):
                observations.append((phi, bead_idx, u, v))

        self._observations = observations
        return observations

    def estimate_bead_positions(self, observations, SOD, SDD, u0, v0, du):
        """
        从观测数据反推每颗珠子的 3D 坐标。

        方法：对每颗珠子的 u(phi) 轨迹做正弦拟合：
            u(phi) = C + A_cos * cos(phi) + A_sin * sin(phi)
        然后：
            x_k = A_cos * du * SOD / SDD
            y_k = A_sin * du * SOD / SDD
            z_k = (k - 2.5) * 10.0

        参数：
            observations: [(phi, bead_idx, u, v), ...]
            SOD, SDD: mm
            u0, v0: 像素坐标（原始分辨率）
            du: 像素间距 mm/pixel

        返回：
            (bead_positions, u0_estimated): bead_positions (6,3) mm, u0_estimated from sinusoidal fit
        """
        bead_positions = np.zeros((6, 3))
        C_values = []

        for k in range(6):
            phi_list = []
            u_list = []
            v_list = []
            for phi, bead_idx, u, v in observations:
                if bead_idx == k:
                    phi_list.append(phi)
                    u_list.append(u)
                    v_list.append(v)

            if len(phi_list) < 3:
                print(
                    f"Warning: bead {k} has only {len(phi_list)} observations, skipping"
                )
                continue

            phi_arr = np.array(phi_list)
            u_arr = np.array(u_list)
            v_arr = np.array(v_list)

            H = np.column_stack(
                [np.ones(len(phi_arr)), np.cos(phi_arr), np.sin(phi_arr)]
            )
            C, A_cos, A_sin = np.linalg.lstsq(H, u_arr, rcond=None)[0]

            C_values.append(C)
            bead_positions[k, 0] = A_cos * du * SOD / SDD
            bead_positions[k, 1] = A_sin * du * SOD / SDD
            mean_v = np.mean(v_arr)
            bead_positions[k, 2] = (v0 - mean_v) * du * SOD / SDD

        u0_estimated = np.mean(C_values)
        return bead_positions, u0_estimated

    def estimate_v0(self, observations, bead_positions, SOD, SDD, v0_initial, du):
        """
        从 v 观测数据中精确估计 v0（探测器光轴交点的 v 坐标）。

        方法：珠子的 z 坐标与 mean_v 呈线性关系：
            mean_v_k = v0_true - z_k * SDD / (SOD * du)
        做线性回归：mean_v = slope * z + intercept
        intercept 就是 v0_true（z=0 时的 v 坐标）。

        参数：
            observations: [(phi, bead_idx, u, v), ...]
            bead_positions: (6, 3) 珠子位置（z 列已由 estimate_bead_positions 估算）
            SOD, SDD, du: 几何参数
            v0_initial: 估算 bead_positions 时使用的 v0（用于第一次迭代）

        返回：
            (v0_data, slope): v0_data 是线性拟合截距，slope 是斜率（用于验证）
        """
        mean_v_list = []
        z_list = []

        for k in range(6):
            v_values = []
            for phi, bead_idx, u, v in observations:
                if bead_idx == k:
                    v_values.append(v)
            if len(v_values) > 0:
                mean_v_list.append(np.mean(v_values))
                z_list.append(bead_positions[k, 2])

        mean_v_arr = np.array(mean_v_list)
        z_arr = np.array(z_list)

        H = np.column_stack([np.ones(len(z_arr)), z_arr])
        intercept, slope = np.linalg.lstsq(H, mean_v_arr, rcond=None)[0]

        theoretical_slope = -SDD / (SOD * du)

        return intercept, slope, theoretical_slope

    def calculate(self, observations=None):
        """
        原有校准逻辑，返回 SOD, SDD, u0, v0, theta。
        """
        if observations is None:
            observations = self.load_img()

        A = np.zeros((6, 4, 2))
        U = np.zeros(6)
        V = np.zeros(6)
        ellipses = []

        bead_points = [[] for _ in range(6)]
        for phi, bead_idx, u, v in observations:
            bead_points[bead_idx].append((u, v))

        for i in range(6):
            points = bead_points[i]
            if len(points) < 3:
                continue
            ellipse = cv2.fitEllipse(np.array(points))
            cv2.ellipse(self.zero_img, ellipse, (122, 122, 122), 1)
            theta = ellipse[2] / 180 * np.pi
            ellipses.append(ellipse)
            cx, cy = ellipse[0][0], ellipse[0][1]
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
            if a < b:
                a, b = b, a
                theta -= np.pi / 2

            cost = np.cos(theta)
            sint = np.sin(theta)
            U[i] = cx
            V[i] = cy
            A[i][3][0] = cx + a * cost
            A[i][3][1] = cy + a * sint
            A[i][2][0] = cx - a * cost
            A[i][2][1] = cy - a * sint
            A[i][0][0] = cx - b * sint
            A[i][0][1] = cy + b * cost
            A[i][1][0] = cx + b * sint
            A[i][1][1] = cy - b * cost
        for t in A:
            cv2.circle(self.zero_img, (int(t[0][0]), int(t[0][1])), 3, (255, 0, 0), 1)
            cv2.circle(self.zero_img, (int(t[1][0]), int(t[1][1])), 3, (0, 255, 0), 1)
            cv2.circle(self.zero_img, (int(t[2][0]), int(t[2][1])), 3, (0, 0, 255), 1)
            cv2.circle(
                self.zero_img, (int(t[3][0]), int(t[3][1])), 3, (255, 255, 255), 1
            )
        X = np.zeros(6)
        Y = np.zeros(6)
        for i in range(6):
            Y[i] = (A[i][0][1] - A[i][1][1]) / ellipses[i][1][1]
            X[i] = ellipses[i][0][1]
        p = np.polyfit(Y[1:], X[1:], 1)
        b1, a1 = p[0], p[1]
        v0 = a1
        SDD = b1
        p = np.polyfit(V[1:], U[1:], 1)
        print("UV P:", p)
        b2, a2 = p[0], p[1]
        u0 = a2 + b2 * v0
        theta = np.arctan(b2) * 180 / np.pi
        SOD = 0
        L = 10
        for i in range(2, 6):
            distance = (U[1] - U[i]) ** 2 + (V[1] - V[i]) ** 2
            SOD += (
                L
                * (i - 1)
                * SDD
                / (
                    distance
                    + ellipses[1][1][1] ** 2
                    + ellipses[i][1][1] ** 2
                    - 2 * ellipses[1][1][1] * ellipses[i][1][1]
                )
                ** 0.5
            )
        SOD /= 6 - 2
        SDD = SDD * self.dpixel
        return round(SOD, 2), round(SDD, 2), round(u0, 2), round(v0, 2), round(theta, 2)

    def calculate_vshift_package(self):
        """
        两阶段校准，输出完整参数包（含 v-shift）。

        Stage-1: 现有 calculate() 获得几何初值 + estimate_bead_positions 获得 u0_est
        Stage-2: joint_optimize() 优化 eta/vc/vs

        返回 dict:
          SOD, SDD, u0_raw, v0_raw, eta, vc_raw, vs_raw,
          rms_init, rms_final, sx, sy,
          u0_recon, v0_recon, vc_recon, vs_recon,
          notes
        """
        observations = self.load_img()

        SOD_cal, SDD_cal, u0_cal, v0_cal, theta_cal = self.calculate(observations)

        du = dv = self.dpixel
        bead_positions, u0_est = self.estimate_bead_positions(
            observations, SOD_cal, SDD_cal, u0_cal, v0_cal, du
        )

        (
            best_eta,
            best_SOD,
            best_SDD,
            best_vc,
            best_vs,
            refined_positions,
            final_rms,
            rms_init,
        ) = joint_optimize(
            observations, SOD_cal, SDD_cal, u0_est, v0_cal, du, dv, bead_positions
        )

        sx = 0.5
        sy = 0.5
        u0_raw = round(u0_cal, 2)
        v0_raw = round(v0_cal, 2)
        vc_raw = round(best_vc, 6)
        vs_raw = round(best_vs, 6)
        eta_raw = round(best_eta, 6)
        SOD_raw = round(best_SOD, 2)
        SDD_raw = round(best_SDD, 2)

        u0_recon = round(u0_raw * sx, 2)
        v0_recon = round(v0_raw * sy, 2)
        vc_recon = round(vc_raw * sy, 6)
        vs_recon = round(vs_raw * sy, 6)

        print(
            f"[CalibResult] SOD={SOD_raw}, SDD={SDD_raw}, "
            f"u0_raw={u0_raw}, v0_raw={v0_raw}, "
            f"eta={eta_raw}, vc_raw={vc_raw}, vs_raw={vs_raw}"
        )
        print(
            f"[CalibResult] vc_recon={vc_recon}, vs_recon={vs_recon}, "
            f"RMS init={rms_init:.4f} -> final={final_rms:.4f}"
        )

        return {
            "SOD": SOD_raw,
            "SDD": SDD_raw,
            "u0_raw": u0_raw,
            "v0_raw": v0_raw,
            "eta": eta_raw,
            "vc_raw": vc_raw,
            "vs_raw": vs_raw,
            "rms_init": round(rms_init, 4),
            "rms_final": round(final_rms, 4),
            "sx": sx,
            "sy": sy,
            "u0_recon": u0_recon,
            "v0_recon": v0_recon,
            "vc_recon": vc_recon,
            "vs_recon": vs_recon,
            "notes": {
                "v_shift_sign_in_conebeam": "use -(vc*cos(phi)+vs*sin(phi))",
                "angles": "phi comes from filename degrees (relative angle), deg->rad",
            },
        }


def reproject(P, phi, SOD, SDD, u0, v0, eta, du, dv, vc=0.0, vs=0.0):
    """
    将3D点P投影到探测器平面，返回像素坐标 (u_proj, v_proj)。
    v0 随角度正弦变化：v0_eff(phi) = v0 + vc*cos(phi) + vs*sin(phi)
    """
    ODD = SDD - SOD

    src = np.array([np.sin(phi) * SOD, -np.cos(phi) * SOD, 0.0])
    det0 = np.array([-np.sin(phi) * ODD, np.cos(phi) * ODD, 0.0])

    u_dir = np.array([np.cos(phi), np.sin(phi), 0.0])
    v_dir = np.array([-eta * np.sin(phi), eta * np.cos(phi), -1.0])

    n_hat = np.cross(u_dir, v_dir)

    ray = P - src

    denom = np.dot(n_hat, ray)
    if np.abs(denom) < 1e-10:
        return None

    t = np.dot(det0 - src, n_hat) / denom
    hit = src + t * ray

    delta = hit - det0
    u_proj = u0 + np.dot(delta, u_dir) / du
    v_proj = (
        v0
        + np.dot(delta, v_dir) / (dv * np.dot(v_dir, v_dir))
        + vc * np.cos(phi)
        + vs * np.sin(phi)
    )

    return (u_proj, v_proj)


def joint_optimize(
    observations, SOD_init, SDD_init, u0_fixed, v0_fixed, du, dv, init_positions
):
    """
    联合优化 eta, SOD, SDD, vc, vs 和珠子 3D 位置。v0 固定。

    参数向量: [eta, SOD, SDD, vc, vs, x0,y0,z0, x1,y1,z1, ..., x5,y5,z5]
    共 5 + 18 = 23 个参数

    参数:
        observations: [(phi, bead_idx, u, v), ...]
        SOD_init, SDD_init: 几何参数初始值
        u0_fixed: u0 固定值（不优化）
        v0_fixed: v0 固定值（不优化）
        du, dv: 像素间距
        init_positions: shape (6, 3), 初始珠子位置

    返回:
        (best_eta, best_SOD, best_SDD, best_vc, best_vs, refined_positions, final_rms, rms_init)
    """
    n_beads = 6
    n_geom = 5  # eta, SOD, SDD, vc, vs
    n_params = n_geom + n_beads * 3

    x0 = np.zeros(n_params)
    x0[0] = 0.0  # eta
    x0[1] = SOD_init  # SOD
    x0[2] = SDD_init  # SDD
    x0[3] = 0.0  # vc
    x0[4] = 0.0  # vs
    for k in range(n_beads):
        x0[n_geom + k * 3 : n_geom + (k + 1) * 3] = init_positions[k]

    lb = np.zeros(n_params)
    ub = np.zeros(n_params)
    lb[0] = -0.1
    ub[0] = 0.1
    lb[1] = 800.0
    ub[1] = 1100.0
    lb[2] = 800.0
    ub[2] = 1200.0
    lb[3] = -30.0
    ub[3] = 30.0
    lb[4] = -30.0
    ub[4] = 30.0
    for k in range(n_beads):
        base = n_geom + k * 3
        for i in range(3):
            lb[base + i] = init_positions[k, i] - 50.0
            ub[base + i] = init_positions[k, i] + 50.0

    def residuals(params):
        eta = params[0]
        SOD = params[1]
        SDD = params[2]
        vc = params[3]
        vs = params[4]
        res = []
        for phi, bead_idx, u_meas, v_meas in observations:
            bead_params = params[n_geom + bead_idx * 3 : n_geom + (bead_idx + 1) * 3]
            P = np.array(bead_params)
            proj = reproject(P, phi, SOD, SDD, u0_fixed, v0_fixed, eta, du, dv, vc, vs)
            if proj is not None:
                res.append(proj[0] - u_meas)
                res.append(proj[1] - v_meas)
            else:
                res.append(0.0)
                res.append(0.0)
        return np.array(res)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=10000,
    )

    best_eta = result.x[0]
    best_SOD = result.x[1]
    best_SDD = result.x[2]
    best_vc = result.x[3]
    best_vs = result.x[4]
    refined_positions = np.zeros((6, 3))
    for k in range(6):
        refined_positions[k] = result.x[n_geom + k * 3 : n_geom + (k + 1) * 3]

    errors = residuals(result.x)
    final_rms = np.sqrt(np.mean(errors**2))
    rms_init = np.sqrt(np.mean(residuals(x0) ** 2))

    return (
        best_eta,
        best_SOD,
        best_SDD,
        best_vc,
        best_vs,
        refined_positions,
        final_rms,
        rms_init,
    )


if __name__ == "__main__":
    proj_path = "/home/foods/pro/data/20260327-jz-1/"

    c = Calibration(proj_path, 0.0748, 6, 1536, 1944)

    print("Loading projections...")
    observations = c.load_img()
    print(f"Total observations: {len(observations)}")

    print("\n=== Calibration (reference) ===")
    SOD_cal, SDD_cal, u0_cal, v0_cal, theta_cal = c.calculate(observations)
    print(
        f"calculate(): SOD={SOD_cal}, SDD={SDD_cal}, u0={u0_cal}, v0={v0_cal}, theta={theta_cal}"
    )

    SOD = 910.7
    SDD = 978.11
    u0 = 751.77
    v0 = 1013.91
    du = dv = 0.0748
    print(f"Using config: SOD={SOD}, SDD={SDD}, u0={u0}, v0={v0}")

    print("\n=== Estimating bead 3D positions ===")
    bead_positions, u0_est = c.estimate_bead_positions(
        observations, SOD, SDD, u0, v0, du
    )
    print(f"u0 from config: {u0}, u0 from data: {u0_est:.2f}")
    print("Estimated bead positions (mm):")
    for k in range(6):
        print(
            f"  bead {k}: x={bead_positions[k, 0]:.2f}, y={bead_positions[k, 1]:.2f}, z={bead_positions[k, 2]:.2f}"
        )

    print("\n=== Estimating v0 from v observations ===")
    v0_data, slope, slope_theory = c.estimate_v0(
        observations, bead_positions, SOD, SDD, v0, du
    )
    print(f"v0 from config: {v0}")
    print(f"v0 from calculate(): {v0_cal}")
    print(f"v0 from data (linear fit): {v0_data:.2f}")
    print(f"slope from data: {slope:.4f}")
    print(f"theoretical slope (-SDD/(SOD*du)): {slope_theory:.4f}")
    print(f"slope error: {(slope - slope_theory) / slope_theory * 100:.2f}%")

    print("\n=== Joint optimization: eta + SOD + SDD + vc + vs + bead positions ===")
    (
        best_eta,
        best_SOD,
        best_SDD,
        best_vc,
        best_vs,
        refined_positions,
        final_rms,
        rms_init,
    ) = joint_optimize(observations, SOD, SDD, u0_est, v0, du, dv, bead_positions)
    vc_amp = np.sqrt(best_vc**2 + best_vs**2)
    vc_phase = np.arctan2(best_vs, best_vc) * 180 / np.pi
    print(f"\n--- Parameter comparison ---")
    print(f"eta:   init=0.000000  ->  opt={best_eta:.6f}")
    print(
        f"SOD:   init={SOD:.2f}  ->  opt={best_SOD:.2f}  (delta={best_SOD - SOD:.2f})"
    )
    print(
        f"SDD:   init={SDD:.2f}  ->  opt={best_SDD:.2f}  (delta={best_SDD - SDD:.2f})"
    )
    print(f"vc:    init=0.000000  ->  opt={best_vc:.6f}")
    print(f"vs:    init=0.000000  ->  opt={best_vs:.6f}")
    print(f"v-shift amplitude: {vc_amp:.4f} px, phase: {vc_phase:.2f} deg")
    print(f"v0:    fixed at {v0} (not optimized)")
    print(f"u0:    fixed at {u0_est:.2f} (not optimized)")
    if best_SDD <= best_SOD:
        print(
            f"WARNING: SDD ({best_SDD:.2f}) <= SOD ({best_SOD:.2f}) - physically impossible!"
        )
    print(
        f"\nRMS: init={rms_init:.4f}  ->  opt={final_rms:.4f}  (improvement={(rms_init - final_rms) / rms_init * 100:.2f}%)"
    )
    print(f"\nRefined bead positions (mm):")
    for k in range(6):
        print(
            f"  bead {k}: x={refined_positions[k, 0]:.2f}, y={refined_positions[k, 1]:.2f}, z={refined_positions[k, 2]:.2f}"
        )
    print(f"\nPer-bead RMS:")
    for k in range(6):
        bead_obs = [(phi, b, u, v) for phi, b, u, v in observations if b == k]
        errors = []
        for phi, _, u_meas, v_meas in bead_obs:
            P = refined_positions[k]
            proj = reproject(
                P,
                phi,
                best_SOD,
                best_SDD,
                u0_est,
                v0,
                best_eta,
                du,
                dv,
                best_vc,
                best_vs,
            )
            if proj is not None:
                errors.append(
                    np.sqrt((proj[0] - u_meas) ** 2 + (proj[1] - v_meas) ** 2)
                )
        if errors:
            print(f"  bead {k}: rms={np.sqrt(np.mean(np.array(errors) ** 2)):.3f} px")
    print(
        f"\nOptimized params: SOD={best_SOD:.2f}, SDD={best_SDD:.2f}, u0={u0_est:.2f}, v0={v0:.2f}, eta={best_eta:.6f}, vc={best_vc:.6f}, vs={best_vs:.6f}"
    )
    vc_recon = best_vc * 0.5
    vs_recon = best_vs * 0.5
    print(f"For conebeam.py: vc_recon={vc_recon:.4f}, vs_recon={vs_recon:.4f}")

    print("\n=== Residual diagnostics ===")
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    diag_dir = "/home/foods/pro/data/20260327-jz-1/diagnostics/"
    os.makedirs(diag_dir, exist_ok=True)

    all_u_res = []
    all_v_res = []

    for k in range(6):
        bead_obs = [(phi, b, u, v) for phi, b, u, v in observations if b == k]
        if len(bead_obs) == 0:
            continue

        rows = []
        u_res_list = []
        v_res_list = []
        total_err_list = []

        for phi, _, u_meas, v_meas in bead_obs:
            P = refined_positions[k]
            proj = reproject(
                P,
                phi,
                best_SOD,
                best_SDD,
                u0_est,
                v0,
                best_eta,
                du,
                dv,
                best_vc,
                best_vs,
            )
            if proj is not None:
                u_proj, v_proj = proj
                u_res = u_proj - u_meas
                v_res = v_proj - v_meas
                total_err = np.sqrt(u_res**2 + v_res**2)
            else:
                u_proj, v_proj = np.nan, np.nan
                u_res, v_res = np.nan, np.nan
                total_err = np.nan
            rows.append((phi, u_meas, v_meas, u_proj, v_proj, u_res, v_res, total_err))
            u_res_list.append(u_res)
            v_res_list.append(v_res)
            total_err_list.append(total_err)
            all_u_res.append(u_res)
            all_v_res.append(v_res)

        df = pd.DataFrame(
            rows,
            columns=[
                "phi",
                "u_meas",
                "v_meas",
                "u_proj",
                "v_proj",
                "u_res",
                "v_res",
                "total_err",
            ],
        )
        csv_path = os.path.join(diag_dir, f"residuals_bead{k}.csv")
        df.to_csv(csv_path, index=False)

        u_res_arr = df["u_res"].values
        v_res_arr = df["v_res"].values
        total_arr = df["total_err"].values

        print(f"\n--- bead {k} ---")
        print(
            f"  u_res: mean={np.nanmean(u_res_arr):.3f}, std={np.nanstd(u_res_arr):.3f}, median={np.nanmedian(u_res_arr):.3f}"
        )
        print(
            f"  v_res: mean={np.nanmean(v_res_arr):.3f}, std={np.nanstd(v_res_arr):.3f}, median={np.nanmedian(v_res_arr):.3f}"
        )
        outlier_count = np.sum(total_arr > 30)
        print(
            f"  total: rms={np.sqrt(np.nanmean(total_arr**2)):.3f}, median_abs={np.nanmedian(total_arr):.3f}, outliers(>30px)={outlier_count}/{len(total_arr)}"
        )

        phi_arr = df["phi"].values
        H_u = np.column_stack([np.ones(len(phi_arr)), np.cos(phi_arr), np.sin(phi_arr)])
        try:
            C_u, A_cos_u, A_sin_u = np.linalg.lstsq(H_u, u_res_arr, rcond=None)[0]
            u_fit = C_u + A_cos_u * np.cos(phi_arr) + A_sin_u * np.sin(phi_arr)
            u_fit_res = u_res_arr - u_fit
            print(
                f"  u_res sin fit: A_cos={A_cos_u:.4f}, A_sin={A_sin_u:.4f}, C={C_u:.4f}, residual_std={np.std(u_fit_res):.4f}"
            )
        except:
            pass

        H_v = np.column_stack([np.ones(len(phi_arr)), np.cos(phi_arr), np.sin(phi_arr)])
        try:
            C_v, A_cos_v, A_sin_v = np.linalg.lstsq(H_v, v_res_arr, rcond=None)[0]
            v_fit = C_v + A_cos_v * np.cos(phi_arr) + A_sin_v * np.sin(phi_arr)
            v_fit_res = v_res_arr - v_fit
            print(
                f"  v_res sin fit: A_cos={A_cos_v:.4f}, A_sin={A_sin_v:.4f}, C={C_v:.4f}, residual_std={np.std(v_fit_res):.4f}"
            )
        except:
            pass

    all_u_res = np.array(all_u_res)
    all_v_res = np.array(all_v_res)

    fig1, axes1 = plt.subplots(2, 6, figsize=(24, 8))
    for k in range(6):
        bead_obs = [(phi, b, u, v) for phi, b, u, v in observations if b == k]
        if len(bead_obs) == 0:
            axes1[0, k].set_title(f"bead {k}")
            axes1[1, k].set_title(f"bead {k}")
            continue
        phi_list = [x[0] for x in bead_obs]
        u_list = [x[2] for x in bead_obs]
        v_list = [x[3] for x in bead_obs]

        u_proj_list = []
        v_proj_list = []
        for phi, _, u_m, v_m in bead_obs:
            P = refined_positions[k]
            proj = reproject(
                P,
                phi,
                best_SOD,
                best_SDD,
                u0_est,
                v0,
                best_eta,
                du,
                dv,
                best_vc,
                best_vs,
            )
            if proj:
                u_proj_list.append(proj[0])
                v_proj_list.append(proj[1])
            else:
                u_proj_list.append(np.nan)
                v_proj_list.append(np.nan)

        u_res_arr = np.array(u_proj_list) - np.array(u_list)
        v_res_arr = np.array(v_proj_list) - np.array(v_list)
        rms = np.sqrt(np.nanmean(u_res_arr**2 + v_res_arr**2))

        axes1[0, k].scatter(phi_list, u_res_arr, alpha=0.3, s=10)
        axes1[0, k].set_title(f"bead {k}, rms={rms:.2f}")
        axes1[0, k].set_xlabel("phi")
        axes1[0, k].axhline(0, color="r", linestyle="--", lw=0.5)
        axes1[1, k].scatter(phi_list, v_res_arr, alpha=0.3, s=10)
        axes1[1, k].set_title(f"bead {k}")
        axes1[1, k].set_xlabel("phi")
        axes1[1, k].axhline(0, color="r", linestyle="--", lw=0.5)
    for k in range(6):
        axes1[0, k].set_ylabel("u_res (px)")
        axes1[1, k].set_ylabel("v_res (px)")
    fig1.suptitle(
        f"Residuals vs angle (SOD={best_SOD:.1f}, SDD={best_SDD:.1f}, eta={best_eta:.6f}, vc={best_vc:.3f}, vs={best_vs:.3f})"
    )
    fig1.tight_layout()
    fig1.savefig(os.path.join(diag_dir, "residuals_vs_angle.png"), dpi=150)
    print(f"\nSaved {os.path.join(diag_dir, 'residuals_vs_angle.png')}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    valid_u = all_u_res[~np.isnan(all_u_res)]
    valid_v = all_v_res[~np.isnan(all_v_res)]
    axes2[0].hist(valid_u, bins=50, alpha=0.7)
    axes2[0].axvline(
        np.mean(valid_u),
        color="r",
        linestyle="--",
        label=f"mean={np.mean(valid_u):.3f}",
    )
    axes2[0].axvline(
        np.std(valid_u), color="g", linestyle="--", label=f"std={np.std(valid_u):.3f}"
    )
    outlier_pct_u = np.sum(np.abs(valid_u) > 30) / len(valid_u) * 100
    axes2[0].set_title(f"u_res histogram (outliers >30px: {outlier_pct_u:.1f}%)")
    axes2[0].set_xlabel("u_res (px)")
    axes2[0].legend()
    axes2[1].hist(valid_v, bins=50, alpha=0.7)
    axes2[1].axvline(
        np.mean(valid_v),
        color="r",
        linestyle="--",
        label=f"mean={np.mean(valid_v):.3f}",
    )
    axes2[1].axvline(
        np.std(valid_v), color="g", linestyle="--", label=f"std={np.std(valid_v):.3f}"
    )
    outlier_pct_v = np.sum(np.abs(valid_v) > 30) / len(valid_v) * 100
    axes2[1].set_title(f"v_res histogram (outliers >30px: {outlier_pct_v:.1f}%)")
    axes2[1].set_xlabel("v_res (px)")
    axes2[1].legend()
    fig2.suptitle("Residual histograms")
    fig2.tight_layout()
    fig2.savefig(os.path.join(diag_dir, "residuals_hist.png"), dpi=150)
    print(f"Saved {os.path.join(diag_dir, 'residuals_hist.png')}")

    print("\n=== Done ===")
