import numpy as np
import pdb
import time
import os
import cv2
import matplotlib.pyplot as plt
import csv

def get_ids(filename):
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f , delimiter=',')
        for row in reader:
            ids[ row[0][0:-4] ] = int( row[1] )
    return ids


def match(img, kernel):
    output = cv2.filter2D(img, -1, kernel)
    return len((output[1:-1,1:-1]==kernel.sum()).nonzero()[0])

cords8 = [[(-1,1)], [(-1,0)], [(-1,-1)], [(0,-1)], [(1,-1)], [(1,0)], [(1,1)], [(0,1)]]

cords16 = [[(-1,1),(-2,2)], [(-1,0),(-2,0)], [(-1,-1),(-2,-2)],
            [(0,-1),(0,-2)], [(1,-1),(2,-2)], [(1,0),(2,0)],
            [(1,1),(2,2)], [(0,1),(0,2)]]

cords32 = [[(-1,1),(-2,2),(-3,3)], [(-1,0),(-2,0),(-3,0)], [(-1,-1),(-2,-2),(-3,-3)],
            [(0,-1),(0,-2),(0,-3)], [(1,-1),(2,-2),(3,-3)], [(1,0),(2,0),(3,0)],
            [(1,1),(2,2),(3,3)], [(0,1),(0,2),(0,3)]]

shape = (5,5)
bank_32 = list()
filters_32 = list()
for cords in cords16:
    filt = np.zeros(shape , dtype = np.uint8)
    for point in cords:
        x = point[0]
        y = point[1]
        filt[shape[0]/2, shape[1]/2] = 1
        filt[(shape[0]/2 + x),(shape[1]/2 + y)] = 1
    filters_32.append(filt)

# 2 junctions
for i in filters_32:
    for j in filters_32:
        if (i!=j).any():
            bank_32.append(i+j)

# 3 junctions
for i in filters_32:
    for j in filters_32:
        for k in filters_32:
            if (i!=j).any() and (j!=k).any() and (i!=k).any():
                bank_32.append(i+j+k)
# 4 junctions
for i in filters_32:
    for j in filters_32:
        for k in filters_32:
            if (i!=j).any() and (j!=k).any() and (i!=k).any():
                for l in filters_32:
                    if (l!=i).any() and (l!=j).any() and (l!=k).any():
                        bank_32.append(i+j+k+l)
# Removing duplicates if any
bank = [bank_32[0]]
for filt in bank_32:
    flag = 0
    for f in bank:
        if (f==filt).all():
            flag = 1
    if not flag:
        bank.append(filt)

# Making center = 1
for filt in bank:
    filt[2,2] = 1
# pdb.set_trace()

data_path = "/home/chrizandr/data/IAM_blocks/"
output_file ="/home/chrizandr/Texture_Analysis/IAM_blocks/con_234_3.csv"
class_labels = "/home/chrizandr/data/writerid_eng.csv"

labels = get_ids(class_labels)

folderlist = os.listdir(data_path)
folderlist.sort()

f = open(output_file,"w")
log = open("featurex.log","w")
print("Starting")
features = list()
labels = list()
for name in folderlist:
    if name[-4:]=='.png':
        print("Processing " + name)
        start_time = time.time()
        img_name = data_path + name
        img = cv2.imread(img_name, 0)
        img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = 1 - img
        feature = list()
        for kernel in bank:
            feature.append(match(img, kernel))
        feature = np.array(feature)
        features.append(feature)
        labels.append(name[0:-4])
        print("--- %s seconds ---" % (time.time() - start_time))

features = np.array(features)
mean = np.mean(features, axis=0)
std = np.mean(features, axis=0)
feature = (features - mean) / std
for i in range(features.shape[0]):
    for j in range(features.shape[1]):
        f.write(str(features[i,j])+',')
    f.write(str(labels[i]) + '\n')
f.close()
pdb.set_trace()
# Put all kernels and match to get the number of points matching the kernel
# Keep all in ascending order of angles
