import cv2
import imutils
import numpy as np

original = cv2.imread("perfect.png")
new = cv2.imread("rotate45.png")
print(np.shape(original))
print(np.shape(new))
original = imutils.resize(original, height = 300)
# new = imutils.resize(new, height = 300)
new =  cv2.resize(new, (original.shape[1], original.shape[0]))
print(np.shape(original))
print(np.shape(new))
cv2.imshow("original",original)
cv2.imshow("new",new)

org_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
new_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(org_gray, new_gray)
cv2.imshow("diff",diff)

retval, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh",thresh)

red_frame = np.ones_like(original,dtype=np.uint8)*np.array([0, 0, 255],np.uint8)
# cv2.imshow("red",red_frame)

red_diff = cv2.bitwise_and(red_frame, red_frame, mask= thresh)
# cv2.imshow("red_diff",red_diff)

final  = cv2.bitwise_or(original,red_diff)
cv2.imshow("final",final)

cv2.waitKey(0)