import numpy as np
import pdb
import time
import os
import cv2

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary


def match(img, kernel):
    output = cv2.filter2D(img, -1, kernel)
    return len((output[1:-1,1:-1]==kernel.sum()).nonzero()[0])


def feature_set(img, bank):
	img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	img = 1 - img
	feature = list()
	for kernel in bank:
		    feature.append(match(img, kernel))
	feature = np.array(feature)
	return feature

'''features = np.array(features)
mean = np.mean(features, axis=0)
std = np.mean(features, axis=0)
feature = (features - mean) / std

for i in range(features.shape[0]):
    for j in range(features.shape[1]):
        f.write(str(features[i,j])+',')
    f.write(str(labels[i]) + '\n')
'''
def filtergen():
	cords32 = [[(-1,1),(-2,2),(-3,3)], [(-1,0),(-2,0),(-3,0)], [(-1,-1),(-2,-2),(-3,-3)],
		    [(0,-1),(0,-2),(0,-3)], [(1,-1),(2,-2),(3,-3)], [(1,0),(2,0),(3,0)],
		    [(1,1),(2,2),(3,3)], [(0,1),(0,2),(0,3)]]

	bank_32 = list()
	filters_32 = list()
	for cords in cords32:
	    filt = np.zeros((7,7) , dtype = np.uint8)
	    for point in cords:
		x = point[0]
		y = point[1]
		filt[3,3] = 1
		filt[(3+x),(3+y)] = 1
	    filters_32.append(filt)

	for i in filters_32:
	    for j in filters_32 + np.zeros((7,7), dtype = np.uint8):
		if (i!=j).any():
		    bank_32.append(i+j)

	for i in filters_32:
	    for j in filters_32:
		for k in filters_32:
		    if (i!=j).any() and (j!=k).any() and (i!=k).any():
			for l in filters_32 + [np.zeros((7,7), dtype=np.uint8)]:
			    if (l!=i).any() and (l!=j).any() and (l!=k).any():
			        bank_32.append(i+j+k+l)

	bank = [bank_32[0]]
	for filt in bank_32:
	    flag = 0
	    for f in bank:
		if (f==filt).all():
		    flag = 1
	    if not flag:
		bank.append(filt)

	for filt in bank:
	    filt[3,3] = 1

	return bank
# Put all kernels and match to get the number of points matching the kernel
# Keep all in ascending order of angles
