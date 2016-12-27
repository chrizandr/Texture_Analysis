import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import pdb
import os
import multiprocessing

# def feature_set(img):
# 	# img = cv2.imread(img_name,0)
# 	#img= np.array([[5,8,1],[5,4,1],[3,7,2]])
# 	m,n = img.shape
# 	f=[0 for x in range(0,256)]
# 	for i in range(1,m-1):
# 		for j in range(1,n-1):
# 			val = 0
# 			lis = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j+1],img[i+1,j+1],img[i+1,j],img[i+1,j-1],img[i,j-1]]
# 			count=0
# 			mean = (sum(lis)+img[i,j])/float(9)
# 			for each in lis:
# 				if each >= img[i,j] :
# 					val+=2**count
# 				count+=1
# 			f[int(val)]+=1
# 	return f

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


data_path = "/home/chris/honours/text_blocks/"
output_file ="output.csv"
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
# Get a list of all the files in teh dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.remove("writerids.csv")
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
