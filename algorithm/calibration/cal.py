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


def reproject(P, phi, SOD, SDD, u0, v0, eta, du, dv):
    """
    将3D点P投影到探测器平面，返回像素坐标 (u_proj, v_proj)。
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
    v_proj = v0 + np.dot(delta, v_dir) / (dv * np.dot(v_dir, v_dir))

    return (u_proj, v_proj)


def joint_optimize(observations, SOD, SDD, u0_data, v0, du, dv, init_positions):
    """
    联合优化 eta 和珠子 3D 位置。

    参数:
        observations: [(phi, bead_idx, u, v), ...]
        SOD, SDD, u0_data, v0, du, dv: 几何参数
        init_positions: shape (6, 3), 初始珠子位置

    返回:
        (best_eta, refined_positions, final_rms)
    """
    n_beads = 6
    n_params = 1 + n_beads * 3  # eta + 6 beads * (x,y,z)

    # 初始参数向量: [eta, x0,y0,z0, x1,y1,z1, ..., x5,y5,z5]
    x0 = np.zeros(n_params)
    x0[0] = 0.0  # eta = 0
    for k in range(n_beads):
        x0[1 + k * 3 : 1 + (k + 1) * 3] = init_positions[k]

    # 参数边界: eta in [-0.1, 0.1], 珠子位置在初始值 ±50mm
    lb = np.zeros(n_params)
    ub = np.zeros(n_params)
    lb[0] = -0.1
    ub[0] = 0.1
    for k in range(n_beads):
        base = 1 + k * 3
        for i in range(3):
            lb[base + i] = init_positions[k, i] - 50.0
            ub[base + i] = init_positions[k, i] + 50.0

    def residuals(params):
        eta = params[0]
        res = []
        for phi, bead_idx, u_meas, v_meas in observations:
            bead_params = params[1 + bead_idx * 3 : 1 + (bead_idx + 1) * 3]
            P = np.array(bead_params)
            proj = reproject(P, phi, SOD, SDD, u0_data, v0, eta, du, dv)
            if proj is not None:
                res.append(proj[0] - u_meas)
                res.append(proj[1] - v_meas)
            else:
                res.append(0.0)
                res.append(0.0)
        return np.array(res)

    result = least_squares(
        residuals, x0, method="trf", bounds=(lb, ub), loss="soft_l1", f_scale=2.0
    )

    best_eta = result.x[0]
    refined_positions = np.zeros((6, 3))
    for k in range(6):
        refined_positions[k] = result.x[1 + k * 3 : 1 + (k + 1) * 3]

    # 计算最终 RMS
    errors = residuals(result.x)
    final_rms = np.sqrt(np.mean(errors**2))

    # 计算 eta=0 时的 RMS
    x0_only_beads = x0.copy()
    x0_only_beads[0] = 0.0
    errors_at_zero = residuals(x0_only_beads)
    rms_at_zero = np.sqrt(np.mean(errors_at_zero**2))

    return best_eta, refined_positions, final_rms, rms_at_zero


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

    print("\n=== Joint optimization: eta + bead positions ===")
    best_eta, refined_positions, final_rms, rms_at_zero = joint_optimize(
        observations, SOD, SDD, u0_est, v0, du, dv, bead_positions
    )
    print(f"eta=0 RMS: {rms_at_zero:.4f} pixels")
    print(f"optimal eta: {best_eta:.6f}")
    print(f"optimal RMS: {final_rms:.4f} pixels")
    if rms_at_zero > 0:
        print(f"improvement: {(rms_at_zero - final_rms) / rms_at_zero * 100:.2f}%")
    print("\nRefined bead positions (mm):")
    for k in range(6):
        print(
            f"  bead {k}: x={refined_positions[k, 0]:.2f}, y={refined_positions[k, 1]:.2f}, z={refined_positions[k, 2]:.2f}"
        )
