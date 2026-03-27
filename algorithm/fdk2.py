import numpy as np
import cv2
import os
import numba as nb
import nibabel as nib

dd = 74.8 / 1000
SOD = 665.188
TM = 1944
TN = 1536


N = 512

dd_x = dd * N / TN
dd_y = dd * N / TM
SOD2 = np.power(SOD, 2)


@nb.jit(nopython=True)
def init(img: np.ndarray):
    for y in range(N):
        for x in range(N):
            # 加权
            a = x - N / 2
            b = y - N / 2
            img[y, x] = (
                img[y, x]
                * SOD
                / np.sqrt(SOD2 + np.power(a * dd_x, 2) + np.power(b * dd_y, 2))
            )
            # 滤波


@nb.jit(nopython=True)
def backproject(img: np.ndarray, i: int, ni: np.ndarray, ni_tmp: np.ndarray):
    sini = np.sin(i * np.pi / 180)
    cosi = np.cos(i * np.pi / 180)
    for k1 in range(N):
        x = dd_x * (k1 - N / 2)
        for k2 in range(N):
            y = dd_x * (k2 - N / 2)
            U = (SOD + x * sini - y * cosi) / SOD
            U2 = np.power(U, 2)
            a = (x * cosi + y * sini) / U
            xx = round(a / dd_x + N / 2)
            for k3 in range(N):
                z = dd_y * (k3 - N / 2)
                b = z / U
                yy = round(b / dd_y + N / 2)
                if yy < 0 or yy >= N or xx < 0 or xx >= N:
                    continue
                pf = img[yy, xx]
                ni[k2, k1, k3] += pf / U2


import time

lasttime = time.time()
ni = np.zeros((N, N, N), dtype=np.uint64)
for i in range(360):
    if os.path.exists(f"../data1/{i}.tif"):
        print(f"../data1/{i}.tif")
        img = cv2.imread(f"../data1/{i}.tif", -1)

        init(img)
        backproject(img, i, ni, ni)
        # 反投影
print("total time:", time.time() - lasttime)
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
