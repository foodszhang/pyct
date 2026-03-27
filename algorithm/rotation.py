import numpy as np
import cv2
TN=1536
TM=1944
import sys
import os
from concurrent.futures import ThreadPoolExecutor
#dark = cv2.imread('../data/dark.tif', -1)
#empty = cv2.imread('../data/empty.tif', -1)

#def rotation(i):
#    if os.path.exists("../data/1208_calibration/p{:04d}.prj".format(i+1)):
#        img = np.fromfile("../data/1208_calibration/p{:04d}.prj".format(i+1), dtype=np.uint16).reshape((TM, TN))
#    else:
#        return
#    max_img = np.max(img)
#    rotation_mat = cv2.getRotationMatrix2D((TN/2,TM/2),0.15,1)
#    img = cv2.warpAffine(img,rotation_mat,(TN,TM), borderValue=int(max_img))
#    img.tofile("../data/1208_r/p{:04d}.prj".format(i+1))
#    print(f"{i+1}.prj")
#pool = ThreadPoolExecutor(max_workers=20) 
#futs = []
#for i in range(360):
#        futs.append(pool.submit(rotation, i))
#        #rotation(0)
#for fut in futs:
#    fut.result()
        
        

#img = np.fromfile("../data/1208_calibration/d",dtype=np.uint16).reshape((TM, TN))
#max_img = np.max(img)
#rotation_mat = cv2.getRotationMatrix2D((TN/2,TM/2),0.15,1)
#img = cv2.warpAffine(img,rotation_mat,(TN,TM), borderValue=int(max_img))
#img.tofile("../data/1208_r/d".format())
#img = np.fromfile("../data/1208_calibration/e",dtype=np.uint16).reshape((TM, TN))
#img = np.fromfile("../data/1214/e",dtype=np.uint16).reshape((TN, TM))
#cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#img = img.astype(np.uint8)
##max_img = np.max(img)
##rotation_mat = cv2.getRotationMatrix2D((TN/2,TM/2),0.15,1)
##img = cv2.warpAffine(img,rotation_mat,(TN,TM), borderValue=int(max_img))
#cv2.imshow("e", img)
#cv2.waitKey(0)
#img.tofile("../data/1208_r/e".format())

img = np.fromfile("../data/1208_r/p0132.prj",dtype=np.uint16).reshape((TM, TN))
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = img.astype(np.uint8)
#max_img = np.max(img)
#rotation_mat = cv2.getRotationMatrix2D((TN/2,TM/2),0.15,1)
#img = cv2.warpAffine(img,rotation_mat,(TN,TM), borderValue=int(max_img))
cv2.imshow("e", img)
cv2.waitKey(0)
img.tofile("../data/1208_r/e".format())
