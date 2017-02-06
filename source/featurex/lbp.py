import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import pdb
import os
import multiprocessing

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

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary


def extract(name):
	global data_path
	# Loop over the files in the dataset
	if name[-4:]=='.png':     # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]

		print("Processing "+ name)
        img_name = data_path + name
        A = feature_set(img_name)

	return (A , name)

for x in range(3,8):
    data_path = "/home/chris/telugu_blocks%d/" %(x)
    output_file ="LBP%d.csv"%(x)

    folderlist = os.listdir(data_path)
    folderlist.sort()
    # Open the output file and the log file in write mode

    f = open(output_file,"w")

    pool = multiprocessing.Pool(4)
    result = pool.map(extract, folderlist)

    for feature in result:
    	A , name = feature
    	for x in A:
    		f.write(str(x)+',')
    	f.write(name[0:-4] + '\n')

    f.close()
