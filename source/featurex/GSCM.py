import numpy as np
from skimage import feature as skfeature
from skimage.filters import threshold_otsu
import cv2
import os
import time
# ------------------------------ Global variables for GSCM features.
distances = [1,2,3,4,5]         # The distances of the various GSCM matrices to be computed
angles = [0,45,90,135]          # The orientations of the various GSCM matrices to be computed
# ------------------------------

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

def getangle(n):
    return (np.float64)(n*np.pi)/180

def feature_set(name,distances,angles):
    img = cv2.imread(name,0)
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for i in range(len(angles)):
        angles[i] = getangle(angles[i])
    features = list()
    x = skfeature.greycomatrix(img,distances,angles,levels=2, symmetric=True)
    for d in range(len(distances)):
        for theta in range(len(angles)):
            features.append(x[0,0,d,theta])
            features.append(x[1,0,d,theta])
            features.append(x[1,1,d,theta])
    return features
data_path = "/home/chris/honours/bangla_blocks/"
output_file ="output.csv"
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
class_labels = "/home/chris/honours/bangla_blocks/writerids.csv"
# Construct a dictionary out of the given file
labels = get_ids(class_labels)
# Get a list of all the files in teh dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.sort()
# Open the output file and the log file in write mode
f = open(output_file,"w")
log = open("featurex.log","w")
# Loop over the files in the dataset
for name in folderlist:
    if name[-4:]=='.tif':     # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        # try:
            print("Processing "+ name)

            start_time = time.time()          # Code for time measurement
            img_name = data_path + name
            # ------------------------------ The feature set function from the appropriately imported feature source is used.
            A = feature_set(img_name,distances,angles)
            # ------------------------------
            for feature in A:
                f.write(str(feature)+',')
            f.write(name[0:-4]+'\n')
            print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
        # except:                         # Log any files that have some error
            # log.write(name+'\n')
log.close()
f.close()
