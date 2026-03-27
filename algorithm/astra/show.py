import numpy as np
import cv2
#ni = np.load("ni_temp.npy")
ni = np.fromfile("../../rec4.raw", dtype=np.float32).reshape((256,256,512))
#ni = np.fromfile("../../data1/11281.raw", dtype=np.float32).reshape((512,512,512))
ni = np.transpose(ni, (2,1, 0))


ni = np.where(ni > 1000, 1000, ni)
ni = np.where(ni < -1000, -1000, ni)
ni = cv2.normalize(ni,  None, 0, 255, cv2.NORM_MINMAX)
ni = ni.astype(np.uint8)

ni.tofile('ni.raw')
print('init')
cv2.imshow("ni", ni[:,:,0])
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:,:,x]))
cv2.imshow("ni2", ni[0,:,:])
cv2.createTrackbar("x", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x,:,:]))
cv2.imshow("ni3", ni[:,0,:])
cv2.createTrackbar("y", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:,x,:]))
cv2.waitKey(0)
cv2.destroyAllWindows()


