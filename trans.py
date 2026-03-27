import cv2
import os
import numpy as np

for f in os.listdir('.'):
    if f.endswith('.tif'):
        print(f)
        ar = cv2.imread(f, -1)
        ar = ar.astype(np.uint16)
        w = 1944
        h = 1536
        newar = ar.reshape(w,h)
        cv2.imwrite(f, newar)
