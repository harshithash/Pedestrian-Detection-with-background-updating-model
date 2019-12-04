#!/usr/bin/env python
import numpy as np
import skfuzzy as fuzz
#import skimage.io as io 
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# load original image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#load the image
img = mpimg.imread('back.png');
img=rgb2gray(img)

# pixel intensities
I = img.reshape((1, -1))
print I
print I.shape
P=I.transpose()
print P.shape

# params
n_centers = 3
fuzziness_degree = 2
error = 0.005
maxiter = 1000

# fuzz c-means clustering
centers, u, u0, d, jm, n_iters, fpc = fuzz.cluster.cmeans(I, c=n_centers, m=fuzziness_degree, error=error, maxiter=maxiter, init=None)
img_clustered = np.argmax(u, axis=0).astype(float)
#print centers
print u.shape
# display clusters
img_clustered.shape = img.shape

imgplot = plt.imshow(img_clustered)
plt.imsave('pro.jpeg',img_clustered)
plt.show()



