import numpy as np
import pdb
import time
import os
import cv2


def match(img, kernel):
    output = cv2.filter2D(img, -1, kernel)
    return len((output[1:-1,1:-1]==kernel.sum()).nonzero()[0])

cords8 = [[(-1,1)], [(-1,0)], [(-1,-1)], [(0,-1)], [(1,-1)], [(1,0)], [(1,1)], [(0,1)]]
filters_8 = list()
for cords in cords8:
    filt = np.zeros((3,3) , dtype = np.uint8)
    for point in cords:
        x = point[0]
        y = point[1]
        filt[(1+x),(1+y)] = 1
    filters_8.append(filt)

# Put all kernels and match to get the number of points matching the kernel
# Keep all in ascending order of angles

print match(img, kernel)
