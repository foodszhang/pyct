import numpy as np
import cv2
import os
import numba as nb
import typing
import sys
import nibabel as nib

SOD = 722.679
TM = 1944
TN = 1536

du = 14.6883
dv = -185.3831
phi = -0.83178


N = 512
dd = 74.8 / 1000
# dd_x = dd * TN / N
# dd_y = dd * TM / N
dd_x = dd * N / TN
dd_y = dd * N / TM
SOD2 = np.power(SOD, 2)

fft = np.fft.fft
fftshift = np.fft.fftshift
ifft = np.fft.ifft
ifftshift = np.fft.ifftshift

sini = np.sin(np.arange(360) * np.pi / 180)
cosi = np.cos(np.arange(360) * np.pi / 180)


def gen_RL_filter() -> np.ndarray:
    """
    生成R-L滤波器
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


def gen_SL_filter() -> np.ndarray:
    """
    生成S-L滤波器
    """
    filter = np.zeros(N)
    for i in range(N):
        x = i - N / 2
        filter[i] = -2 / ((np.pi * dd_x) ** 2 * (4 * x**2 - 1))

    return filter


@nb.jit(nopython=True)
def init(img: np.ndarray, filter: np.ndarray, aa: np.ndarray, bb: np.ndarray):
    """
    加权 && 滤波
    """

    weight = SOD / (np.sqrt(SOD**2 + aa**2 + bb**2))
    img = img * weight
    # for y in range(N):
    #    img[y,:] = np.convolve(img[y,:], filter, mode='same')
    return img


# Bilinear Interpolation
@nb.jit(nopython=True)
def biliner_interpolation(img: np.ndarray, x: float, y: float) -> float:
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1
    if x2 >= N:
        x2 = N - 1
    if y2 >= N:
        y2 = N - 1
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return img[y2, x2]
    return (
        img[y1, x1] * (x2 - x) * (y2 - y)
        + img[y1, x2] * (x - x1) * (y2 - y)
        + img[y2, x1] * (x2 - x) * (y - y1)
        + img[y2, x2] * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1))


@nb.jit(nopython=True, parallel=True)
def backproject(img: np.ndarray, i: int, ni: np.ndarray):
    sinii = sini[i]
    cosii = cosi[i]
    for k1 in nb.prange(N):
        x = dd_x * (k1 - N / 2)
        for k2 in range(N):
            y = dd_x * (k2 - N / 2)
            U = (SOD + x * sinii - y * cosii) / SOD
            U2 = np.power(U, 2)
            a = (x * cosii + y * sinii) / U
            # 偏移校准
            # a = a + du
            xx = a / dd_x + N / 2
            for k3 in range(N):
                z = dd_y * (k3 - N / 2)
                b = z / U
                yy = b / dd_y + N / 2
                xx = round(xx)
                yy = round(yy)
                if xx < 0:
                    xx = 0
                if xx >= N:
                    xx = N - 1
                if yy < 0:
                    yy = 0
                if yy >= N:
                    yy = N - 1
                ni[k1, k2, k3] += img[yy, xx] / U2
                # ni[k2,k1,k3] += pf / U2


import time

lasttime = time.time()
ni = np.zeros((N, N, N), dtype=np.uint64)
ni_tmp = np.zeros((N, N, N), dtype=np.uint64)
for i in range(360):
    if os.path.exists(f"../data/alway1/{i}.tif"):
        print(f"../data/3000-1-1.56-3.00-360/{i}.tif")
        img = cv2.imread(f"../data/alway1/{i}.tif", -1)
        # img = cv2.warpAffine(img, movement, (N, N), borderValue=6890)
        img = cv2.resize(img, (N, N), img)
        r_l_filter = gen_RL_filter()
        s_l_filter = gen_SL_filter()
        a = (np.arange(N) - N / 2) * dd_x
        b = (np.arange(N) - N / 2) * dd_y
        aa, bb = np.meshgrid(a, b)
        img = init(img, r_l_filter, aa, bb)
        backproject(img, i, ni)

print("over")
ni = cv2.normalize(ni, None, 0, 255, cv2.NORM_MINMAX)
ni = ni.astype(np.uint8)
cv2.imshow("ni", ni[:, :, 0])
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:, :, x]))
cv2.imshow("ni2", ni[0, :, :])
cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x, :, :]))
cv2.imshow("ni3", ni[:, 0, :])
cv2.createTrackbar("x", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:, x, :]))
cv2.waitKey(0)
cv2.destroyAllWindows()
# 反投影
print("total time:", time.time() - lasttime)
nii_out = nib.Nifti1Image(ni, np.eye(4))
nib.save(nii_out, "ni2.nii.gz")
