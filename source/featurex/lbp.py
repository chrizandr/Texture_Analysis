import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import pdb
import os
import multiprocessing
def feature_set(img_name):
	img = cv2.imread(img_name,0)
	#img= np.array([[5,8,1],[5,4,1],[3,7,2]])
	m,n = img.shape
	img2 = np.ones(img.shape)
	f=[0 for x in range(0,256)]
	for i in range(1,m-1):
		for j in range(1,n-1):
			val = 0
			lis = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j+1],img[i+1,j+1],img[i+1,j],img[i+1,j-1],img[i,j-1]]
			count=0
			# mean = (sum(lis)+img[i,j])/float(9)
			for each in lis:
				if img[i,j] < mean :
					val+=2**count
				count+=1
			#img2[i,j]=val
			f[int(val)]+=1
	return f

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary


def extract(name):
	global labels
	# Loop over the files in the dataset
	if name[-4:]=='.png':     # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        #try:
		print("Processing "+ name)
	        label = labels[name[0:-4]]
	        #start_time = time.time()          # Code for time measurement
	        img_name = data_path + name
	        # ------------------------------ The feature set function from the appropriately imported feature source is used.
		print("Extracting features")
	        A = feature_set(img_name)

		# print("Writing to file")
	    #     for feature in A:
	    #         f.write(str(feature)+',')
	    #     f.write(label)
        #print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
    #except:                         # Log any files that have some error
    #    log.write(name+'\n')
	return (name,A)
data_path = "/home/sanny/honours/isi_work/text_blocks/"
output_file ="output.csv"
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
class_labels = "/home/sanny/honours/isi_work/text_blocks/writerids.csv"
# Construct a dictionary out of the given file
# Get a list of all the files in teh dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.sort()
# Open the output file and the log file in write mode
f = open(output_file,"w")
log = open("featurex.log","w")
labels = get_ids(class_labels)
pool = multiprocessing.Pool(4)
features = pool.map(extract, folderlist[0:4])
features = sorted(features, key=lambda x: x[0])
for feature in features:
	name , A = feature
	label = labels[name[0:-4]]
	for x in A:
		f.write(str(x)+',')
	f.write(label)
log.close()
f.close()
