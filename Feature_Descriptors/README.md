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

[opencv implement](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
## HOG(Histogram of Oriented Gradients ): another image descriptor
### Introduction
Typically, a feature descriptor converts an image of size height x width x 3 (channels ) to a feature vector / array of length n. In the case of the HOG feature descriptor, the input image is of size 64 x 128 x 3 and the output feature vector is of length 3780.

Keep in mind that HOG descriptor can be calculated for other sizes, but in this post I am sticking to numbers presented in the original paper so you can easily understand the concept with one concrete example.

This all sounds good, but what is “useful” and what is “extraneous” ? To define “useful”, we need to know what is it “useful” for ? Clearly, the feature vector is not useful for the purpose of viewing the image. But, it is very useful for tasks like image recognition and object detection. The feature vector produced by these algorithms when fed into an image classification algorithms like Support Vector Machine (SVM) produce good results.

In the HOG feature descriptor, the distribution ( histograms ) of directions of gradients ( oriented gradients ) are used as features. Gradients ( x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes ) and we know that edges and corners pack in a lot more information about object shape than flat regions.

### How to calculate Histogram of Oriented Gradients ?
In this section, we will go into the details of calculating the HOG feature descriptor. To illustrate each step, we will use a patch of an image.

#### Step 1 : Preprocessing
As mentioned earlier HOG feature descriptor used for pedestrian detection is calculated on a 64×128 patch of an image. Of course, an image may be of any size. Typically patches at multiple scales are analyzed at many image locations. The only constraint is that the patches being analyzed have a fixed aspect ratio. In our case, the patches need to have an aspect ratio of 1:2. For example, they can be 100×200, 128×256, or 1000×2000 but not 101×205.



- Find robust feature set that allows object form to be discriminated.
- Challenges：
- Wide range of pose and large variations in appearances
- Cluttered backgrounds under different illumination
- “Speed” for mobile vision
## Difference between HoG and SIFT
- HoG is usually used to describe entire images. SIFT is used for key point matching.
- SIFT histrograms are oriented towards the dominant gradient. HoG is not.
- HoG gradients are normalized using neighborhood bins.
- SIFT descriptors use varying scales to compute multiple descriptors
# Coding exercise
## Application: Panorama
