import cv2
import numpy as np
image = cv2.imread('../data/123-2/0.tif', -1)
dark = cv2.imread('../data/123-2/dark.tif', -1)
empty = cv2.imread('../data/123-2/empty.tif', -1)
#image = cv2.imread('../data/alway1/20.tif')
#dark = cv2.imread('../data/dark.tif')
#empty = cv2.imread('../data/empty.tif')
max_dark = np.max(dark)
empty = np.where(empty <= max_dark, max_dark+1,empty)
new_image = (image - dark) / (empty - dark)
print(np.max(new_image), np.min(new_image))
cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX)

new_image = new_image.astype(np.uint8)
print(np.max(new_image), np.min(new_image))
cv2.imshow("ni", new_image)
#print(new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


