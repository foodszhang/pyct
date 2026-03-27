import numpy as np
import cv2
#ni = np.load("ni_temp.npy")
#ni = np.fromfile("ni2.raw", dtype=np.uint8).reshape((512,512,512))
ni = np.fromfile("rec3.raw", dtype=np.uint8).reshape((512,512,512))

print('init')
cv2.imshow("ni", ni[:,:,0].T)
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:,:,x].T))
cv2.imshow("ni2", ni[0,:,:].T)
cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x,:,:].T))
cv2.imshow("ni3", ni[:,0,:].T)
cv2.createTrackbar("y", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:,x,:].T))
cv2.waitKey(0)
cv2.destroyAllWindows()


