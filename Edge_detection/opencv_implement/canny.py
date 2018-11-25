import cv2
import numpy as np

img = cv2.imread('./road.jpg', 0) # read the gray image
cv2.imwrite('canny_img.jpg', cv2.Canny(img, 200, 300))
cv2.imshow('canny edge detection', cv2.imread('canny_img.jpg'))
cv2.waitKey()
cv2.desdroyAllWindows()