import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from blocks import *
import multiprocessing

def get_base_texture(components , max_height , shape):
    constant_x = 0
    components = [x*255 for x in components]
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

# Path for data and output
dataset = "/home/chrizandr/data/IAM_hand/"
# All the files in the dataset
files = os.listdir(dataset)
#files = ["a-68.tif","a-58.tif","a-59.tif","a-49.tif","a-43.tif","a-48.tif","a-14.tif","a-12.tif","a-18.tif","a-68.tif","a-31.tif","a-32.tif","a-23.tif"]
def extract(name):
    # Path for data and output
    dataset = "/home/chrizandr/data/IAM_hand/"
    output = "/home/chrizandr/data/IAM_blocks/"
    # Check if the files are images
    if name[-4:] == ".png":
        print("Processing image : "+ name)
        img = cv2.imread(dataset + name , 0)
        img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = img[5:-5,5:-5]
        components = get_connected_components(img)
        components = refine(components,250)
        comps = refine2(components , img)
        print "Writing blocks"
        for component in comps:
            blocks = get_base_texture(component, img.shape[0], 250)
            index = comps.index(component)
            # pdb.set_trace()
            print index,len(blocks)
            for i in range(len(blocks)):
                cv2.imwrite(output  + name[0:-4] + '-' + str(i) + ".png" , blocks[i])
        print "Done"


pool = multiprocessing.Pool(5)
result = pool.map(extract, files)
pdb.set_trace()
