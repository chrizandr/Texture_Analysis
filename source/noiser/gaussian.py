import numpy as np
import cv2
import os
import pdb
import matplotlib.pyplot as plt
data_path = "/home/chrizandr/data/writing_segment/"
output_path ="/home/chrizandr/data/telugu_ng_"
class_labels = "/home/chrizandr/data/writerids.csv"

files = os.listdir(data_path)
files.sort()
noises = list()

for i in range(1,6):
    s = np.random.normal( 0, i, (5,5)) * 255
    s = s.reshape(1,25)
    noises.append(s)

for s in range(1,6):
    print "sigma = ", s
    output_folder = output_path + str(s) + '/'
    for f in files:
        if f[-4:] == '.png':
            print "Processing - ", f
            img = cv2.imread(data_path + f, 0)
            nimg = img.astype(np.float64)
            noise = np.random.normal(0, s, img.shape).astype(np.float64)
            noise = ( (noise - noise.min() ) / noise.max() ) * (255/3)
            output = np.ones(img.shape) * 255
            output = np.minimum(output, nimg + noise).astype(np.uint8)
            print output_folder + f
            cv2.imwrite(output_folder + f , output)
    print "Finished iteration"
