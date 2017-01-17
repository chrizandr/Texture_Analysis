from skimage.feature import local_binary_pattern,multiblock_lbp
import cv2
import numpy as np


def feature_set(img_name):
    img = cv2.imread(img_name,0)
    m = img.shape[0]
    n = img.shape[1]
    img = local_binary_pattern(img,8,1,'default')
    img = img.astype(np.uint8)
    feature=[0 for i in range(0,256)]
    for i in range(0,m):
        for j in range(0,n):
            feature[img[i,j]]+=1
    return feature
