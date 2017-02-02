import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import multiprocessing
from blocks import *

def find_position(flag , shape):
    for i in range(flag.shape[0] - shape[0]):
        for j in range(flag.shape[1] - shape[1]):
            if flag[i,j] != 1:
                if 1 not in flag[ i : i+shape[0] , j : j+shape[1] ]:
                    return (i,j)
    return -1

def get_text_blocks(components):
    blocks = list()
    flag = np.zeros((300,300))
    block = np.ones((300,300)) * 255
    print("Getting blocks")
    for component in components:
        if component.shape[0] > 300:
            print "block left"
            continue
        if component.shape[1] > 300:
            iterations = (component.shape[1]/300) + 1
            start = 0
            end = component.shape[1] / iterations
            for i in range(iterations):
                t1 = component[:,start:end]
                p1 = find_position(flag , t1.shape)
                if p1 == -1:
                    blocks.append(block)
                    block = np.ones((300,300)) * 255
                    flag = np.zeros((300,300))
                    p1 = (0,0)
                block[p1[0]:p1[0]+t1.shape[0] , p1[1]:p1[1]+t1.shape[1]] = t1
                flag[p1[0]:p1[0]+t1.shape[0] , p1[1]:p1[1]+t1.shape[1]] = 1
                start = end
                end = end + (component.shape[1] / iterations)
                if end > component.shape[1]:
                    end = None
            continue

        position = find_position(flag, component.shape)
        if position == -1:
            blocks.append(block)
            block = np.ones((300,300)) * 255
            flag = np.zeros((300,300))
            position = (0,0)
        block[position[0]:position[0]+component.shape[0] , position[1]:position[1]+component.shape[1]] = component
        flag[position[0]:position[0]+component.shape[0] , position[1]:position[1]+component.shape[1]] = 1
    return blocks

def get_ids():
    f = open("/home/chris/honours/bangla_seg/writerids.csv","r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

def extract(name):
    ids = get_ids()
    # ids["test"] = -1
    writers = list()
    if name[-4:]==".tif":          # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        print("Processing "+ name)
        label = ids[name[0:-4]]
        img = cv2.imread("/home/chris/honours/bangla_seg/" + name , 0)
        components = get_connected_components(img)
        components = refine(components)
        blocks = get_text_blocks(components)
        count = 0
        print("Writing blocks")
        for block in blocks:
            x = cv2.imwrite("/home/chris/honours/bangla_blocks/" + name[0:-4] + str(count) + ".tif" ,block)
            writers.append((name[0:-4]+str(count),label))
            count += 1
    return writers


# Optional main function
data_path = "/home/chris/honours/bangla_seg/"              # Path of the original dataset
output_path = "/home/chris/honours/bangla_blocks/"            # Path of the output folder
writers = open("/home/chris/honours/bangla_blocks/writerids.csv" , "w")
# Get a list of all the files in the dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.remove("writerids.csv")
folderlist.sort()
# pdb.set_trace()
# Open the output file in write mode
print("Starting........")


# img = cv2.imread("test.png" , 0)
# components = get_connected_components(img)
# components = refine(components)
# # pdb.set_trace()
# for comp in components:
#     plt.imshow(comp,'gray')
#     plt.show()
# pdb.set_trace()
# extract("g06-042o.png")

pool = multiprocessing.Pool(5)
result = pool.map(extract, folderlist)
for page in result:
    for blocks in page:
        writers.write(blocks[0] + blocks[1] )
        writers.close()
