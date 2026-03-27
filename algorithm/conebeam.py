import numpy as np
import cv2
import numba as nb
from typing import Optional, Callable
import sys

sini = np.sin(np.arange(360) * np.pi / 180)
cosi = np.cos(np.arange(360) * np.pi / 180)
sin = np.sin
cos = np.cos
atan = np.arctan
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
sinc = np.sinc
sqrt = np.sqrt
real = np.real
ceil = np.ceil
log2 = np.log2
pi = np.pi



@nb.jit(nopython=True, parallel=True)
def _backproject(
    img: np.ndarray,
    i: int,
    result_voxel: np.ndarray,
    N: int,
    SOD: float,
    SDD: float,
    dd_x: float,
    dd_y: float,
    du: float,
    dv: float,
    dd: float,
    TM: int,
    TN: int,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    a_start: float,
    b_start: float,
    a_delta: float,
    b_delta: float,
):
    sinii = sini[i]
    cosii = cosi[i]
    w = SOD/SDD

    for k1 in nb.prange(N):
        x = X[k1]

        for k2 in range(N):
            y = Y[k2]
            t = x * cosii + y * sinii
            s = -x * sinii + y * cosii
            InterpX = (SOD * t) / (SOD - s)
            InterpW = (SOD ** 2) / ((SOD - s) ** 2)
            InterpX = (InterpX - a_start) / a_delta
            for k3 in range(N):
                z = Z[k3]
                InterpY = (SOD * z) / (SOD - s)
                InterpY = (InterpY - b_start) / b_delta
                vq = interpolation(img, InterpX, InterpY, TN, TM)
                #用k1, k2, k3的顺序可以极大的提高性能
                #result_voxel[k1, k2, k3] += pf / U2 * 2 * np.pi
                result_voxel[k1, k2, k3] += vq * InterpW 


#@nb.jit(nopython=True)
def _init(
    img: np.ndarray,
    TN: int,
    TM: int,
    aa: np.ndarray,
    bb: np.ndarray,
    SOD: float,

    filter: np.ndarray,
):
    """
    加权 && 滤波
    """
    ZeroPaddedLength = int(2 ** (ceil(log2(2 * (TN - 1)))))
    weight = SOD / (np.sqrt(SOD**2 + aa**2 + bb**2))
    img = img * weight
    for y in range(TM):
        tmp = real(ifft(
            ifftshift(filter * fftshift(fft(img[y, :], ZeroPaddedLength)))))
        img[y, :] = tmp[0:TN]
    return img


@nb.jit(nopython=True)
def interpolation(img: np.ndarray, x: float, y: float, N: int, M: int):
    x = round(x)
    y = round(y)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= N:
        x = N - 1
    if y >= M:
        y = M - 1
    return img[y, x]


class ConeBeam:
    N = 512

    def __init__(self, SOD: float, dd: float, TN: int, TM: int, du: float,dv:float, phi: float, SDD: float):
        self.SOD = SOD
        self.dd = dd
        self.TN = TN
        self.TM = TM
        self.du = du
        self.dv = dv
        self.dd_x = dd * TN / ConeBeam.N
        self.dd_y = dd * TM / ConeBeam.N
        self.SOD2 = SOD**2
        self.SDD = SDD
        self.result_voxel = np.zeros((ConeBeam.N, ConeBeam.N, ConeBeam.N), np.float64)
        self.phi = phi
        fov = 2.0 * SOD * sin(atan(self.dd * self.TM / 2.0 / (self.SDD)))
        fovz = 2.0 * SOD * sin(atan(self.dd * self.TN / 2.0 / self.SDD))

        self.x = np.linspace(-fov / 2.0, fov / 2.0, self.N)
        self.y = np.linspace(-fov / 2.0, fov / 2.0, self.N)
        self.z = np.linspace(-fovz / 2.0, fovz / 2.0, self.N)

    @staticmethod
    def Filter(N, pixel_size, FilterType, cutoff):
        '''
        TO DO: Ram-Lak filter implementation
                   Argument for name of filter
        '''
        if cutoff > 0.5 or cutoff < 0:
            raise Exception('Cutoff have to be pose between 0 and 0.5')
        x = np.arange(0, N) - (N - 1) / 2
        h = np.zeros(len(x))
        h[np.where(x == 0)] = 1 / (8 * pixel_size ** 2)
        odds = np.where(x % 2 == 1)
        h[odds] = -0.5 / (pi * pixel_size * x[odds]) ** 2
        h = h[0:-1]
        filter = abs(fftshift(fft(h))) * 2
        w = 2 * pi * x[0:-1] / (N - 1)
        if FilterType == 'ram-lak':
            pass  # Do nothing
        elif FilterType == 'shepp-logan':
            zero = np.where(w == 0)
            tmp = filter[zero]
            filter = filter * sin(w / (2 * cutoff)) / (w / (2 * cutoff))
            filter[zero] = tmp * sin(w[zero] / (2 * cutoff))
        elif FilterType == 'cosine':
            filter = filter * cos(w / (2 * cutoff))
        elif FilterType == 'hamming':
            filter = filter * (0.54 + 0.46 * (cos(w / cutoff)))
        elif FilterType == 'hann':
            filter = filter * (0.5 + 0.5 * cos(w / cutoff))

        filter[np.where(abs(w) > pi * cutoff)] = 0
        return filter

    @staticmethod
    @nb.jit(nopython=True)
    def gen_filter(
        dd_x: float, N: int, filter_type: Optional[str] = "RL"
    ) -> np.ndarray:
        """
        生成滤波器
        """
        if filter_type == "RL":
            N = 51
            filter = np.zeros(N)
            for i in range(N):
                x = int(i - N // 2)
                if x == 0:
                    filter[i] = 1 / (4 * dd_x**2)
                elif x % 2 == 0:
                    filter[i] = 0
                else:
                    filter[i] = -1 / ((np.pi * x * dd_x) ** 2)

            return filter
        else:
            filter = np.zeros(N)
            for i in range(N):
                x = i - N // 2
                filter[i] = -2 / ((np.pi * dd_x) ** 2 * (4 * x**2 - 1))

            return filter

    def init(self, img: np.ndarray, filter: np.ndarray):
        """
        加权 && 滤波
        """
        a = (np.arange(self.TN) - self.TN / 2) * self.dd
        b = (np.arange(self.TM) - self.TM / 2) * self.dd
        a  = a * self.SOD / self.SDD
        b  = b * self.SOD / self.SDD
        self.a = a
        self.b = b
        aa, bb = np.meshgrid(a, b)
        return _init(
            img=img,
            aa=aa,
            bb=bb,
            SOD=self.SOD,
            filter=filter,
            TN=self.TN,
            TM=self.TM,
        )

    def backproject(self, img: np.ndarray, i: int):
        return _backproject(
            img=img,
            i=i,
            result_voxel=self.result_voxel,
            N=ConeBeam.N,
            SOD=self.SOD,
            SDD=self.SDD,
            dd_x=self.dd_x,
            dd_y=self.dd_y,
            du=self.du,
            dv=self.dv,
            dd=self.dd,
            TM=self.TM,
            TN=self.TN,
            X=self.x,
            Y=self.y,
            Z=self.z,
            a_start=self.a[0],
            b_start=self.b[0],
            a_delta=self.a[1] - self.a[0],
            b_delta=self.b[1] - self.b[0],
        )


if __name__ == "__main__":
    import os
    import time

    #fdk = ConeBeam(SOD=665.188, dd=0.0748, TN=1536, TM=1944, du=14.6883, phi=-0.83178, SDD=721.49)
    fdk = ConeBeam(SOD=722.679, dd=0.0748, TN=1536, TM=1944, du=6.35,dv=-159.47, phi=-0.83178, SDD=778.676)
    #filter = fdk.gen_filter(fdk.dd, fdk.TN)
    ZeroPaddedLength = int(2 ** (ceil(log2(2 * (fdk.TN - 1)))))
    filter = ConeBeam.Filter(
        ZeroPaddedLength + 1, fdk.dd * fdk.SOD / (fdk.SDD), 'hamming', 0.3)
    for i in range(360):
        if os.path.exists(f"../data/alway/{i}.tif"):
            print(f"{i}.tif")
            img = cv2.imread(f"../data/alway/{i}.tif", -1)
            time1 = time.time()
        #if os.path.exists(f"../data/alway1/{i}.tif"):
        #    print(f"{i}.tif")
        #    img = cv2.imread(f"../data/alway1/{i}.tif", -1)
        #    time1 = time.time()
       # if os.path.exists("../data/1214/P{:04d}.prj".format(i)):
       # #    print(f"{i}.tif")
       # #    img = cv2.imread(f"../data/alway1/{i}.tif", -1)
       # #    time1 = time.time()
       #     img = np.fromfile("../data/1214/P{:04d}.prj".format(i), dtype=np.uint16).reshape((fdk.TM, fdk.TN))

            #进行高斯模糊
            img = fdk.init(img, filter)
            #cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            #img = img.astype(np.uint8)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)
            #sys.exit(0)
            time1 = time.time()
            fdk.backproject(img, i)
            print('backproject', time.time() - time1)
            del img

    cv2.normalize(fdk.result_voxel, fdk.result_voxel, 0, 255, cv2.NORM_MINMAX)
    ni = fdk.result_voxel.astype(np.uint8)
    cv2.imshow("ni", ni[:, :, 0])
    cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:, :, x]))
    cv2.imshow("ni2", ni[0, :, :])
    cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x, :, :]))
    cv2.imshow("ni3", ni[:, 0, :])
    cv2.createTrackbar("x", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:, x, :]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 反投影
    ni.tofile("ni2.raw")
