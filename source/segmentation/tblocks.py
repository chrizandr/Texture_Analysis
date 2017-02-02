import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from blocks import *
import multiprocessing

def get_base_texture(components , max_height , shape):
    constant_x = 0
    comps =list()
    for each in components:
        cmy,cmx = tuple( [int(x) for x in ndimage.measurements.center_of_mass(each)] )
        block = np.ones((max_height, each.shape[1])) * 255
        # try:
        block[max_height/2 - cmy : each.shape[0] - cmy + max_height/2 , :] = each
        comps.append(block)
        # except ValueError:
        #     plt.imshow(each, 'gray')
        #     plt.show()

    lines = list()
    line = comps[0]
    for i in range(1,len(comps)):
        if line.shape[1] + comps[i].shape[1] >= shape:
            if line.shape[1] > shape:
                line = comps[i]
                print "Omitting component ---------"
            else:
                lines.append(line)
                line = comps[i]
        else:
            line = np.concatenate((line, comps[i]), axis=1)
    lines = [strip_white(np.concatenate( (line, 255 * np.ones((line.shape[0], shape - line.shape[1])) ) , axis=1)) for line in lines]
    blocks = list()
    block = lines[0]
    for i in range(1,len(lines)):
        if block.shape[0] + lines[i].shape[0] > shape:
            blocks.append(block)
            block = lines[i]
            continue
        block = np.concatenate((block, lines[i]), axis=0)
    return blocks

# Path for the data
dataset = "/home/chris/telugu_hand/"
output = "/home/chris/telugu_blocks/"
# Class labels for the files
labels = get_ids("/home/chris/telugu/writerids.csv")
# All the files in the dataset
files = os.listdir(dataset)

def extract(name):
    dataset = "/home/chris/telugu_hand/"
    output = "/home/chris/telugu_blocks/"
    if name[-4:] == ".png":
        print("Processing image : "+ name)
        img = cv2.imread(dataset + name , 0)
        components = get_connected_components(img)
        components = refine(components,250)
        blocks = get_base_texture(components, img.shape[0], 250)
        return blocks
        print "Writing blocks"
        for i in range(len(blocks)):
            cv2.imwrite(output + name[0:-4] + '-' + str(i) + ".png" , blocks[i])
        print "Done"

b = extract("c-30.png")
for a in b:
    plt.imshow(a,'gray')
    plt.show()

# Need to improve refine function
# Check for c-30.png

# pool = multiprocessing.Pool(5)
# result = pool.map(extract, files)
