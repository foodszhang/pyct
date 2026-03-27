
#view_Num = 360; # 视角
#num = 6; # 钢珠数目 
#w = 1536; # 投影数据宽
#h = 1944;  #% 投影数据高
#dpixel =0.0748; #mm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor 
import os
import sys
import numba as nb



@nb.jit(nopython=True)
def circle_threshold(img, circle):
    max = 0
    min = 255
    for i in range(int(circle[1] - circle[2])-2, int(circle[1] + circle[2])+2):
        for j in range(int(circle[0] - circle[2])-1, int(circle[0] + circle[2])+2):
            distance = np.sqrt((i - circle[1]) ** 2 + (j - circle[0]) ** 2)
            if distance <= circle[2]:
                if img[i, j] > max:
                    max = img[i, j]
                if img[i, j] < min:
                    min = img[i,j]
    return min, max



def hough_circles(img):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=0,maxRadius=0)
    if circles is None:
        return []
    return circles[0]


class Calibration:
    def __init__(self, proj_path,dpixel,num, w, h):
        self.proj_path = proj_path
        self.dpixel = dpixel
        self.num = num
        self.w = w
        self.h = h
        self.zero_img = np.zeros((h, w, 3), dtype=np.uint8)

    def read_circle(self, i):
        if not os.path.exists(os.path.join(self.proj_path, f'{i}.tif')):
            return None
        img = cv2.imread(os.path.join(self.proj_path, f'{i}.tif'), -1)
        img = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.medianBlur(img,5)
        simg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        simg = simg.astype(np.uint8)
        circles = hough_circles(simg)
        if len(circles) != 6 and len(circles) != 0:
            threshold = circle_threshold(simg, circles[0])
            th = np.where((simg <= threshold[1]) & (simg >= threshold[0]), simg, 255)
            circles = hough_circles(th)
            if len(circles) != 6:
                for  circle in circles:
                    cv2.circle(th, (int(circle[0]), int(circle[1])), int(circle[2]), 125, 1)
                cv2.imwrite(f'c{i}.tif', th)
                return None
        return [(x[0], x[1]) for x in circles]

    def load_img(self):
        pool = ThreadPoolExecutor(max_workers=50)
        fut_list = []
        ellipse_points = [[],[],[],[],[],[]]

        for i in range(0,360):
            fut = pool.submit(self.read_circle, i)
            fut_list.append(fut)

        for fut in fut_list:
            points = fut.result()
            if points is None:
                continue
            points = sorted(points, key=lambda x: x[1])
            for i,p in enumerate(points):
                ellipse_points[i].append(p)
        return ellipse_points


    def calculate(self):
        A = np.zeros((6, 4, 2))
        U = np.zeros(6)
        V = np.zeros(6)
        ellipses =[]
        ellipse_points = self.load_img()
        for i in range(6):
            ellipse = cv2.fitEllipse(np.array(ellipse_points[i]))
            cv2.ellipse(self.zero_img, ellipse, (122,122,122), 1)
            theta = ellipse[2] / 180 * np.pi
            ellipses.append(ellipse)
            cx,cy = ellipse[0][0], ellipse[0][1]
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
            if a < b:
                a, b = b,a
                theta -= np.pi / 2

            cost = np.cos(theta)
            sint = np.sin(theta)
            U[i] = cx
            V[i] = cy
            A[i][3][0] = cx+a*cost
            A[i][3][1] = cy+a*sint
            A[i][2][0] = cx-a*cost
            A[i][2][1] = cy-a*sint
            A[i][0][0] = cx-b*sint
            A[i][0][1] = cy+b*cost
            A[i][1][0] = cx+b*sint
            A[i][1][1] = cy-b*cost
        for t in A:
            cv2.circle(self.zero_img, (int(t[0][0]), int(t[0][1])), 3, (255,0,0), 1)
            cv2.circle(self.zero_img, (int(t[1][0]), int(t[1][1])), 3, (0,255,0), 1)
            cv2.circle(self.zero_img, (int(t[2][0]), int(t[2][1])), 3, (0,0,255), 1)
            cv2.circle(self.zero_img, (int(t[3][0]), int(t[3][1])), 3, (255,255,255), 1)
        X =np.zeros(6)
        Y = np.zeros(6)
        for i in range(6):
            Y[i] = (A[i][0][1] - A[i][1][1])/ ellipses[i][1][1]
            X[i] = ellipses[i][0][1]
        p = np.polyfit(Y[1:], X[1:], 1)
        b1,a1 = p[0],p[1]
        v0 = a1
        SDD = b1
        p = np.polyfit(V[1:], U[1:], 1)
        print('UV P:',p)
        b2,a2 = p[0],p[1]
        u0 = a2 + b2*v0
        theta = np.arctan(b2) * 180 / np.pi
        SOD = 0
        L = 10
        for i in range(2,6):
            distance = (U[1]-U[i])**2+(V[1]-V[i])**2;
            SOD += L * (i-1) * SDD/(distance+ellipses[1][1][1]**2+ellipses[i][1][1]**2-2*ellipses[1][1][1]*ellipses[i][1][1])**0.5;
        SOD /= 6-2
        SDD = SDD * self.dpixel
        return round(SOD, 2), round(SDD, 2), round(u0, 2), round(v0, 2), round(theta, 2)
if __name__ == '__main__':
    c = Calibration('../../data/123-3', 0.0748, 6, 1536, 1944)
    SOD, SDD, du, dv, theta = c.calculate()
    print(SOD, SDD, du, dv, theta)



