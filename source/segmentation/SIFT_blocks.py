import cv2
import numpy as np
import matplotlib.pyplot as plt

import pdb

img = cv2.imread("test_block.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img,None)

nimg = cv2.drawKeypoints(img, kp , img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
pdb.set_trace()
