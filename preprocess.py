import os
import cv2
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
from skimage.util import invert
import pdb
import numpy as np

#extract strokes
def match(img, kernel):
    output = cv2.filter2D(img, -1, kernel)
    print output
    return len((output[1:-1,1:-1]==kernel.sum()).nonzero()[0])

cords8 = [[(-1,1)], [(-1,0)], [(-1,-1)], [(0,-1)], [(1,-1)], [(1,0)], [(1,1)], [(0,1)]]

cords16 = [[(-1,1),(-2,2)], [(-1,0),(-2,0)], [(-1,-1),(-2,-2)],
            [(0,-1),(0,-2)], [(1,-1),(2,-2)], [(1,0),(2,0)],
            [(1,1),(2,2)], [(0,1),(0,2)]]

cords32 = [[(-1,1),(-2,2),(-3,3)], [(-1,0),(-2,0),(-3,0)], [(-1,-1),(-2,-2),(-3,-3)],
            [(0,-1),(0,-2),(0,-3)], [(1,-1),(2,-2),(3,-3)], [(1,0),(2,0),(3,0)],
            [(1,1),(2,2),(3,3)], [(0,1),(0,2),(0,3)]]

shape = (7,7)
bank_32 = list()
filters_32 = list()
for cords in cords32:
    filt = np.zeros(shape , dtype = np.uint8)
    for point in cords:
        x = point[0]
        y = point[1]
        filt[shape[0]/2, shape[1]/2] = 1
        filt[(shape[0]/2 + x),(shape[1]/2 + y)] = 1
    filters_32.append(filt)

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
    filt[3,3] = 1

pdb.set_trace()

data_path = "/home/sanny/Documents/clustering_model/data/"
output_file ="/home/chrizandr/Texture_Analysis/noise/Features/telugu_ng_5.csv"

folderlist = os.listdir(data_path)
folderlist.sort()

features = list()
for name in folderlist:
    if name[-4:]=='.png':
        print("Processing "+ name)

        img_name = data_path + name
        img = cv2.imread(img_name, 0)
        gray_img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        gray_img= 1-gray_img
        skeleton = skeletonize_3d(gray_img)
        skeleton = cv2.threshold(skeleton , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        skeleton = 1 - skeleton
        #img2 = thin(gray_img)
        plt.imshow(skeleton, cmap="gray")
        plt.show()
        feature = list()
        for kernel in bank:
            output = cv2.filter2D(skeleton, -1, kernel)
            plt.imshow(output, cmap="gray")
            plt.show()
