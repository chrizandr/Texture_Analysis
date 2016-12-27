from skimage.feature import local_binary_pattern,multiblock_lbp
import cv2
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

def get_ids():
	f = open("writerids.csv","r")
	dictionary = {}
	for line in f:
		line = line.split(",")
		dictionary[line[0]] = line[1]
	return dictionary

def feature_set(img_name):
    img = cv2.imread(img_name,0)
    m = img.shape[0]
    n = img.shape[1]
    img = local_binary_pattern(img,8,1,'default')
    # pdb.set_trace()
	img = img.astype(np.uint8)
    feature=[0 for i in range(0,256)]
    for i in range(0,m):
        for j in range(0,n):
            feature[img[i,j]]+=1
    return feature

fet = feature_set("test2.png")
pdb.set_trace()
