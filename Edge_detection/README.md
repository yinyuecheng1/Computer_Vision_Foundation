# Edge Detection Syllabus
## Edge detection
- Goal: Identify sudden changes(discontinuities) in an image.
- Intuitively, most semantic and shape information from the image can be encoded in the edges.
- More compact than pixels.
- Meaning: Extract information, recognize objects; Recover geometry and viewpoint.


## Image Gradients
- In fact: Discrete Derivative

## A simple edge detector
- Smooth first.
- Get the derivative.
### Sobel edge detector
- Uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives
- One for horizontal changes, and one for vertical
![sobel](https://github.com/yinyuecheng1/Computer_Vision_Foundation/raw/master/Edge_detection/snapshot/sobel.png)
### Canny edge detector
#### Introduction
- This is probably the most widely used edge detector in computer vision.
- Theoretical model: step-edges corrupted by additive Gaussian noise
- Canny has shown that the first derivative of the Gaussian closely approximates the operator that optimizes the product of signal-to-noise ratio and localization.
#### Implementation
- Suppress Noise
- Compute gradient magnitude and direction
- Apply Non-Maximum Suppression(Assures minimal response)
- Use hysteresis and connectivity analysis to detect edges
## Hough Transform
### Intro to Hough transform
- The Hough transform (HT) can be used to detect lines.
- It was introduced in 1962 (Hough 1962) and first used to find lines in images a decade later (Duda 1972).
- The goal is to find the location of lines in images.
- `Caveat`: Hough transform can detect lines, circles and other structures ONLY if their parametric equation is known.
- It can give robust detection under noise and partial occlusion.
### Prior to Hough transform
- Assume that we have performed some edge detection, and a thresholding of the edge magnitude image.
- Thus, we have some pixels that may partially describe the boundary of some objects.

### Detecting lines using Hough transform
- We wish to find sets of pixels that make up straight lines.
- Consider a point of known coordinates (xi;yi).There are many lines passing through the point (xi ,yi ).
- Straight lines that pass that point have the form yi= a*xi + b
- Common to them is that they satisfy the equation for some set of parameters(a, b)

### Detecting lines using Hough transform
- Two points (x1, y1) and(x2 y2) define a line in the (x, y) plane.
- These two points give rise to two different lines in (a,b) space.
- In (a,b) space these lines will intersect in a point (a’ b’)
- All points on the line defined by (x1, y1) and (x2 , y2) in (x, y) space will parameterize lines that intersect in (a’, b’) in (a,b) space.
