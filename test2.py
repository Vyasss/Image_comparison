import cv2
import imutils 
import numpy as np
from skimage.metrics import structural_similarity

before = cv2.imread('perfect.png')
after = cv2.imread('shifted_left_right.png')

before = imutils.resize(before, height = 300)
# after = imutils.resize(after, height = 300)
after =  cv2.resize(after, (before.shape[1], before.shape[0]))

cv2.imshow("before",before)
cv2.imshow("after",after)

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
# print("Image similarity", score)

diff = (diff * 255).astype("uint8")
cv2.imshow("diff",diff)
retval, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thresh",thresh)

red_frame = np.ones_like(before,dtype=np.uint8)*np.array([0, 0, 255],np.uint8)
cv2.imshow("red",red_frame)

red_diff = cv2.bitwise_and(red_frame, red_frame, mask= thresh)
cv2.imshow("red_diff",red_diff)

final  = cv2.bitwise_or(before,red_diff)
cv2.imshow("final",final)

cv2.waitKey(0)