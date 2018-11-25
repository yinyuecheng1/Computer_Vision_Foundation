import cv2
import numpy as np

img = cv2.imread('./road.jpg', 0) # read the gray image
edges = cv2.Canny(img, 200, 300)
minLineLength = 20
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv2.imshow('edges', edges)
cv2.imshow('lines', img)

cv2.waitKey()
cv2.desdroyAllWindows()