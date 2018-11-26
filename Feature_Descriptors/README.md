# Syllabus

## Scale invariant region selection
### Automatic scale selection
### Difference-of-Gaussian (DoG) detector
## SIFT(Scale-invariant feature transform): an image region descriptor
[Wikipedia](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
- Detect keypoint first(DoG).
- A 16x16 neighbourhood around the keypoint is taken. It is devided into 16 sub-blocks of 4x4 size. For each sub-block, 8 bin orientation histogram is created. So a total of 128 bin values are available. It is represented as a vector to form keypoint descriptor. In addition to this, several measures are taken to achieve robustness against illumination changes, rotation etc.

![sift](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/1.png)

- opencv implementation
```
import cv2
import numpy as np

img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)
```
- 8 orientation bins per histogram, and a 4x4 histogram array, yields 8 x 4x4 = 128 numbers.
- So a SIFT descriptor is a length 128 vector, which is invariant to rotation (because we rotated the descriptor) and scale (because
we worked with the scaled image from DoG)
- We can compare each vector from image A to each vector from image B to find matching keypoints! Euclidean “distance” between descriptor vectors gives a good measure of keypoint similarity.

[opencv implement]https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
## HOG: another image descriptor
# Coding exercise
## Application: Panorama
