import numpy as np
import cv2
TN=1536
TM=1944
img = np.fromfile("../data/1214/e", dtype=np.uint16).reshape((TM, TN))
for i in range(0,TN//2):
    img[:,i], img[:,TN-i-1] = img[:,TN-i-1].copy(), img[:,i].copy()
#cv2.imwrite("../data/1214/e.tif",img.T)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

