import cv2
import pdb
import matplotlib.pyplot as plt
from skimage.feature import multiblock_lbp
import numpy as np
from skimage.transform import integral_image

def Multi_lbp(name):
    img = cv2.imread(name , 0)
    m,n = img.shape
    bsize = 9
    mid = bsize/2
    for i in range(bsize , m-bsize):
        for j in range(bsize , n-bsize):
            block = img[ i-mid:i+mid , j-mid:j+mid ]
            int_img = integral_image(test_img)
