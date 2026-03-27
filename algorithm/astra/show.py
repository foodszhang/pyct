import numpy as np
import cv2
import nibabel as nib

# ni = np.load("ni_temp.npy")
ni = nib.load("../../rec4.nii.gz").get_fdata()
# ni = np.fromfile("../../data1/11281.raw", dtype=np.float32).reshape((512,512,512))
ni = np.transpose(ni, (2, 1, 0))


ni = np.where(ni > 1000, 1000, ni)
ni = np.where(ni < -1000, -1000, ni)
ni = cv2.normalize(ni, None, 0, 255, cv2.NORM_MINMAX)
ni = ni.astype(np.uint8)

nii_out = nib.Nifti1Image(ni, np.eye(4))
nib.save(nii_out, "ni.nii.gz")
print("init")
cv2.imshow("ni", ni[:, :, 0])
cv2.createTrackbar("z", "ni", 0, 511, lambda x: cv2.imshow("ni", ni[:, :, x]))
cv2.imshow("ni2", ni[0, :, :])
cv2.createTrackbar("x", "ni2", 0, 511, lambda x: cv2.imshow("ni2", ni[x, :, :]))
cv2.imshow("ni3", ni[:, 0, :])
cv2.createTrackbar("y", "ni3", 0, 511, lambda x: cv2.imshow("ni3", ni[:, x, :]))
cv2.waitKey(0)
cv2.destroyAllWindows()
