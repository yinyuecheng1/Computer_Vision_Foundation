# Syllabus
## Image Histogram

Histogram of an image provides `the frequency of the brightness(intensity) value` in the image.

```
def histogram(im):
  h = np.zeros(255)
  for row in im.shape[0]:
    for col in im.shape[1]:
      val = im[row, col]
      h[val] += 1
```

## Systems and Filters
### Filtering
Forming a new image whose pixel values are transformed from original pixel values.
Goal is to extract useful information from images, or transform images into another domain where we can modify/enhance image properties.
- Features (edges, corners, blobs…)
- super-resolution; in-painting; de-noising

###  linear shift invariant system
- Details in the CS131 slides.
## Convolution and correlation

### Convolution
- X-flip the convolution filter.
- Y-flip the convolution filter.
- Innerproduct with the origin image.
![convolution](https://github.com/yinyuecheng1/Computer_Vision_Foundation/Convolutions/snapshot/convolution.png)
