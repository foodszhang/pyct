import numpy as np
import cv2
#ni = np.load("ni_temp.npy")
#ni = np.fromfile("rec2.raw", dtype=np.uint8).reshape((512,512,512))
ni = np.fromfile("../../data1/12281.raw", dtype=np.float32).reshape((512,512,512))

ni = cv2.normalize(ni,  None, 0, 255, cv2.NORM_MINMAX)
ni = ni.astype(np.uint8)
ni = np.where(ni < 200, 200, ni)
ni = cv2.normalize(ni,  None, 0, 255, cv2.NORM_MINMAX)
ni.tofile('ni2.raw')
print('init')
cv2.imshow("ni", np.flip(ni[:,:,0], axis=0))
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", np.flip(ni[:,:,x], axis=0)))
cv2.imshow("ni2", np.flip(ni[0,:,:], axis=0))
cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", np.flip(ni[x,:,:], axis=0)))
cv2.imshow("ni3", np.flip(ni[:,0,:], axis=0))
cv2.createTrackbar("y", "ni3", 0, 511, lambda x: cv2.imshow("ni3", np.flip(ni[:,x,:], axis=0)))
#cv2.imshow("ni", ni[:,:,0])
#cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:,:,x]))
#cv2.imshow("ni2", ni[0,:,:])
#cv2.createTrackbar("y", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x,:,:]))
#cv2.imshow("ni3", ni[:,0,:])
#cv2.createTrackbar("y", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:,x,:]))
cv2.waitKey(0)
cv2.destroyAllWindows()


