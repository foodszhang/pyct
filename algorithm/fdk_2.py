import numpy as np
import cv2
import os
import numba as nb
import typing
import nibabel as nib

dd = 74.8 / 1000
SOD = 722.679
TM = 1944
TN = 1536
du = 14.6883

R = SOD
N = 512

dd_x = dd * TN / N
dd_y = dd * TM / N
SOD2 = np.power(SOD, 2)


def gen_filter() -> np.ndarray:
    """
    生成滤波器
    """
    filter = np.zeros(N)
    for i in range(N):
        x = i - N / 2
        if x == 0:
            filter[i] = 1 / (4 * dd_x**2)
        elif x % 2 == 0:
            filter[i] = 0
        else:
            filter[i] = -1 / (np.pi * x * dd_x) ** 2
    return filter


@nb.jit(nopython=True)
def init(img: np.ndarray, filter: np.ndarray, xx: np.ndarray, yy: np.ndarray):
    """
    加权 && 滤波
    """
    weight = SOD / (np.sqrt(SOD**2 + aa**2 + bb**2))
    img = img * weight
    # for y in range(N):
    #    img[y,:] = np.convolve(img[y,:], filter, mode='same')
    return img


@nb.jit(nopython=True, parallel=True)
def backproject(img: np.ndarray, i: int, ni: np.ndarray):
    sini = np.sin(i * np.pi / 180)
    cosi = np.cos(i * np.pi / 180)
    for k1 in nb.prange(N):
        z = (k1 - N / 2) * dd_y
        for k2 in range(N):
            x = (k2 - N / 2) * dd_x
            for k3 in range(N):
                y = (k3 - N / 2) * dd_x
                U = R - x * cosi - y * sini
                AU = R + x * cosi + y * sini
                a = R * (-x * sini + y * cosi) / U
                a += du
                b = R * z / U
                aa = a / dd_x + N / 2
                bb = N / 2 + b / dd_x
                aa = round(aa)
                bb = round(bb)
                if aa < 0:
                    aa = 0
                if aa >= N:
                    aa = N - 1
                if bb < 0:
                    bb = 0
                if bb >= N:
                    bb = N - 1
                ni[k1, k2, k3] += img[bb, aa] * SOD2 / AU / AU
    return ni


import time

lasttime = time.time()
ni = np.zeros((N, N, N), dtype=np.uint64)

# 约束
# 虚拟探测器中心为旋转中心
# x,y,z 为重建结果坐标 是真实世界坐标, 原点在虚拟探测器中心, X轴朝内，Z轴朝上，Y轴朝左， 遵循右手定则
# a,b 为重建结果在虚拟探测器上的投影坐标, 原点在虚拟探测器中心
# i 为投影角度
# R = SOD = 放射源到虚拟探测器中心的距离

for i in range(360):
    if os.path.exists(f"../data/alway1/{i}.tif"):
        print(f"../data1/{i}.tif")
        img = cv2.imread(f"../data/alway1/{i}.tif", -1)
        r_l_filter = gen_filter()
        x = (np.arange(N) - N / 2) * dd_x
        y = (N / 2 - np.arange(N)) * dd_y
        aa, bb = np.meshgrid(x, y)
        img = init(img, r_l_filter, aa, bb)
        backproject(img, i, ni)

print("over")
for i in range(512):
    ni[:, :, i] = cv2.normalize(ni[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
ni = ni.astype(np.uint8)
cv2.imshow("ni", ni[:, :, 0])
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:, :, x]))
cv2.imshow("ni2", ni[0, :, :])
cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x, :, :]))
cv2.imshow("ni3", ni[:, 0, :])
cv2.createTrackbar("x", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:, x, :]))
cv2.waitKey(0)
cv2.destroyAllWindows()
nii_out = nib.Nifti1Image(ni, np.eye(4))
nib.save(nii_out, "ni.nii.gz")
# 反投影
print("total time:", time.time() - lasttime)
