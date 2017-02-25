import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

def get_ids(name):
    f = open(name,"r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = int(line[1])
    return dictionary

# Path for the data
dataset = "/home/chrizandr/Documents/"
output = "/home/chrizandr/data/telugu_hand2/"
# Class labels for the files
labels = get_ids("/home/chrizandr/data/writerids.csv")
# All the files in the dataset
files = os.listdir(dataset)

for each in files:
    if each[-4:] == ".png":
        print("Processing imge : ", each)
        img = cv2.imread(dataset + each , 0)
        bimg = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        secth = bimg[:,(img.shape[1]/2 - 100) : (img.shape[1]/2 + 100)]
        sectv = bimg[(img.shape[0]/2 - 100) : (img.shape[0]/2 + 100), : ]
        indices = (np.sum(1-secth,axis =1) > (secth.shape[1] - 20) ).nonzero()[0]
        hor_max = indices.max()-5
        hor_min = indices.min()+5
        indices = (np.sum(1-sectv,axis =0) > (sectv.shape[0] - 20) ).nonzero()[0]
        ver_max = indices.max()-5
        ver_min = indices.min()+5
        nimg = img[hor_min:hor_max , ver_min:ver_max]
        img = img[hor_min:hor_max , ver_min:ver_max]
        cv2.imwrite(output + each , img)
        print("Output written")
