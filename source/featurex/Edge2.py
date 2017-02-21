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

cords8 = [[(-1,1)], [(-1,0)], [(-1,-1)], [(0,-1)], [(1,-1)], [(1,0)], [(1,1)], [(0,1)]]

cords16 = [[(-1,1),(-2,2)], [(-1,0),(-2,0)], [(-1,-1),(-2,-2)],
            [(0,-1),(0,-2)], [(1,-1),(2,-2)], [(1,0),(2,0)],
            [(1,1),(2,2)], [(0,1),(0,2)]]

filters_8 = list()
for cords in cords8:
    filt = np.zeros((3,3) , dtype = np.uint8)
    for point in cords:
        x = point[0]
        y = point[1]
        filt[1,1] = 1
        filt[(1+x),(1+y)] = 1
    filters_8.append(filt)

filters_16 = list()
for cords in cords16:
    filt = np.zeros((5,5) , dtype = np.uint8)
    for point in cords:
        x = point[0]
        y = point[1]
        filt[2,2] = 1
        filt[(2+x),(2+y)] = 1
    filters_16.append(filt)
pdb.set_trace()

bank_8 = list()
bank_16 = list()
for i in filters_8:
    for j in filters_8:
        for k in filters_8:
            if (i!=j).any() and (j!=k).any() and (i!=k).any():
                bank_8.append(i+j+k)

for i in filters_16:
    for j in filters_16:
        for k in filters_16:
            if (i!=j).any() and (j!=k).any() and (i!=k).any():
                bank_16.append(i+j+k)


data_path = "/home/chris/telugu_blocks4/"
output_file ="features8.csv"
class_labels = "/home/chris/writerids.csv"

labels = get_ids(class_labels)

folderlist = os.listdir(data_path)
folderlist.sort()

f = open(output_file,"w")
log = open("featurex.log","w")

for name in folderlist:
    if name[-4:]=='.png':
        print("Processing "+ name)

        start_time = time.time()

        img_name = data_path + name
        img = cv2.imread(img_name, 0)
        img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        feature = list()
        for kernel in bank_16:
            feature.append(match(img,kernel))

        feature = np.array(feature)
        feature = feature - feature.mean()
        for feat in feature:
            f.write(str(feat)+',')
        f.write(name[0:-4] + '\n')
        print("--- %s seconds ---" % (time.time() - start_time))





# Put all kernels and match to get the number of points matching the kernel
# Keep all in ascending order of angles

print match(img, kernel)
