# Syllabus
## Image Histogram

Histogram of an image provides the frequency of the brightness(intensity) value in the image.
'''
def histogram(im):
  h = np.zeros(255)
  for row in im.shape[0]:
    for col in im.shape[1]:
      val = im[row, col]
      h[val] += 1
'''
