# -------------------------------------------------
import cv2
import numpy as np
import os
import time
# -------------------------------------------------

def get_Gabor_features(name,kernels):
    img = cv2.imread(name,0)
    features = list()
    for kernel in kernels:
        rimg = cv2.filter2D(img,cv2.CV_32F,kernel)
        features.append(rimg.mean())
        features.append(rimg.std())
    return features

def getangle(n):
    return (np.float64)(n*np.pi)/180

def make_Gabor_kernels(sigma,angles,frequencies):
    kernels = list()
    for f in frequencies:
        for theta in angles:
            kernels.append(cv2.getGaborKernel((30,30),sigma,getangle(theta),f,1))
    return kernels


def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary
# ------------------------------ Global variables for Gabor features.
sigma = 2                       # Variance of the Gaussian Kernel
frequencies = [4,8,16,32]       # The scales of the different Gabor filters to be used
angles = [0,45,90,135]
# ------------------------------
kernels = make_Gabor_kernels(sigma,angles,frequencies)
data_path = "/home/chris/honours/text_blocks/"
output_file ="output.csv"
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
class_labels = "/home/chris/honours/text_blocks/writerids.csv"
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
    if name[-4:]=='.png':     # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        try:
            print("Processing "+ name)

            label = labels[name[0:-4]]

            start_time = time.time()          # Code for time measurement
            img_name = data_path + name
            # ------------------------------ The feature set function from the appropriately imported feature source is used.
            A = get_Gabor_features(img_name,kernels)
            # ------------------------------
            for feature in A:
                f.write(str(feature)+',')
            f.write(label)
            print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
        except:                         # Log any files that have some error
            log.write(name+'\n')
log.close()
f.close()

# ---------------------------------------------------------------------------------------------------------
