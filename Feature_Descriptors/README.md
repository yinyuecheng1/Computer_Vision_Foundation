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

To illustrate this point I have shown a large image of size 720×475. We have selected a patch of size 100×200 for calculating our HOG feature descriptor. This patch is cropped out of an image and resized to 64×128. Now we are ready to calculate the HOG descriptor for this image patch.

![hog](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog.png)
#### Step 2 : Calculate the Gradient Images
To calculate a HOG descriptor, we need to first calculate the horizontal and vertical gradients; after all, we want to calculate the histogram of gradients. This is easily achieved by filtering the image with the following kernels.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog1.png)
We can also achieve the same results, by using Sobel operator in OpenCV with kernel size 1.
```
// C++ gradient calculation. 
// Read image
Mat img = imread("bolt.png");
img.convertTo(img, CV_32F, 1/255.0);
 
// Calculate gradients gx, gy
Mat gx, gy; 
Sobel(img, gx, CV_32F, 1, 0, 1);
Sobel(img, gy, CV_32F, 0, 1, 1);
```
```
# Python gradient calculation 
 
# Read image
im = cv2.imread('bolt.png')
im = np.float32(im) / 255.0
 
# Calculate gradient 
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
```
Next, we can find the magnitude and direction of gradient using the following formula.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog2.png)

If you are using OpenCV, the calculation can be done using the function cartToPolar as shown below.
```
// C++ Calculate gradient magnitude and direction (in degrees)
Mat mag, angle; 
cartToPolar(gx, gy, mag, angle, 1); 
The same code in python looks like this.
```
```
# Python Calculate gradient magnitude and direction ( in degrees ) 
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
The figure below shows the gradients.
```
The figure below shows the gradients.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog3.png)

Notice, the x-gradient fires on vertical lines and the y-gradient fires on horizontal lines. The magnitude of gradient fires where ever there is a sharp change in intensity. None of them fire when the region is smooth. I have deliberately left out the image showing the direction of gradient because direction shown as an image does not convey much.

The gradient image removed a lot of non-essential information ( e.g. constant colored background ), but highlighted outlines. In other words, you can look at the gradient image and still easily say there is a person in the picture.

At every pixel, the gradient has a magnitude and a direction. For color images, the gradients of the three channels are evaluated ( as shown in the figure above ). The magnitude of gradient at a pixel is the maximum of the magnitude of gradients of the three channels, and the angle is the angle corresponding to the maximum gradient.

#### Step 3 : Calculate Histogram of Gradients in 8×8 cells
In this step, the image is divided into 8×8 cells and a histogram of gradients is calculated for each 8×8 cells.

We will learn about the histograms in a moment, but before we go there let us first understand why we have divided the image into 8×8 cells. One of the important reasons to use a feature descriptor to describe a patch of an image is that it provides a compact representation. 
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog4.png)

An 8×8 image patch contains 8x8x3 = 192 pixel values. The gradient of this patch contains 2 values ( magnitude and direction ) per pixel which adds up to 8x8x2 = 128 numbers. By the end of this section we will see how these 128 numbers are represented using a 9-bin histogram which can be stored as an array of 9 numbers. Not only is the representation more compact, calculating a histogram over a patch makes this represenation more robust to noise. Individual graidents may have noise, but a histogram over 8×8 patch makes the representation much less sensitive to noise.

But why 8×8 patch ? Why not 32×32 ? It is a design choice informed by the scale of features we are looking for. HOG was used for pedestrian detection initially. 8×8 cells in a photo of a pedestrian scaled to 64×128 are big enough to capture interesting features ( e.g. the face, the top of the head etc. ).

The histogram is essentially a vector ( or an array ) of 9 bins ( numbers ) corresponding to angles 0, 20, 40, 60 … 160.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog5.png)

If you are a beginner in computer vision, the image in the center is very informative. It shows the patch of the image overlaid with arrows showing the gradient — the arrow shows the direction of gradient and its length shows the magnitude. Notice how the direction of arrows points to the direction of change in intensity and the magnitude shows how big the difference is.

On the right, we see the raw numbers representing the gradients in the 8×8 cells with one minor difference — the angles are between 0 and 180 degrees instead of 0 to 360 degrees. These are called “unsigned” gradients because a gradient and it’s negative are represented by the same numbers. In other words, a gradient arrow and the one 180 degrees opposite to it are considered the same. But, why not use the 0 – 360 degrees ? Empirically it has been shown that unsigned gradients work better than signed gradients for pedestrian detection. Some implementations of HOG will allow you to specify if you want to use signed gradients.

The next step is to create a histogram of gradients in these 8×8 cells. The histogram contains 9 bins corresponding to angles 0, 20, 40 … 160.

The following figure illustrates the process. We are looking at magnitude and direction of the gradient of the same 8×8 patch as in the previous figure. A bin is selected based on the direction, and the vote ( the value that goes into the bin ) is selected based on the magnitude. Let’s first focus on the pixel encircled in blue. It has an angle ( direction ) of 80 degrees and magnitude of 2. So it adds 2 to the 5th bin. The gradient at the pixel encircled using red has an angle of 10 degrees and magnitude of 4. Since 10 degrees is half way between 0 and 20, the vote by the pixel splits evenly into the two bins.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog6.png)

There is one more detail to be aware of. If the angle is greater than 160 degrees, it is between 160 and 180, and we know the angle wraps around making 0 and 180 equivalent. So in the example below, the pixel with angle 165 degrees contributes proportionally to the 0 degree bin and the 160 degree bin.
![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog7.png)

The contributions of all the pixels in the 8×8 cells are added up to create the 9-bin histogram. For the patch above, it looks like this

![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog8.png)



### Step 4 : 16×16 Block Normalization

![hog1](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Feature_Descriptors/snapshot/hog9.png)

Now that we know how to normalize a vector, you may be tempted to think that while calculating HOG you can simply normalize the 9×1 histogram the same way we normalized the 3×1 vector above. It is not a bad idea, but a better idea is to normalize over a bigger sized block of 16×16. A 16×16 block has 4 histograms which can be concatenated to form a 36 x 1 element vector and it can be normalized just the way a 3×1 vector is normalized. The window is then moved by 8 pixels ( see animation ) and a normalized 36×1 vector is calculated over this window and the process is repeated.
![hog1](https://github.com/yinyuecheng1/interview-computer-vision/raw/master/Image/png/7_8.gif)

interview-computer-vision/Image/png/

### Step 5 : Calculate the HOG feature vector
To calculate the final feature vector for the entire image patch, the 36×1 vectors are concatenated into one giant vector. What is the size of this vector ? Let us calculate

- How many positions of the 16×16 blocks do we have ? There are 7 horizontal and 15 vertical positions making a total of 7 x 15 = 105 positions.
- Each 16×16 block is represented by a 36×1 vector. So when we concatenate them all into one gaint vector we obtain a 36×105 = 3780 dimensional vector.





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
