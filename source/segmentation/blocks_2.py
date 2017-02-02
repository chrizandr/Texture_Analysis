import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import multiprocessing
from scipy import ndimage
import math
from blocks import *

def projection(a,flag):
    nrows,ncols = a.shape
    ver_filter = np.ones((ncols,1))
    hor_filter = np.ones((1,nrows))
    if flag==1 :
        proj = np.dot(a,ver_filter)
        proj = proj.T
        # Dimensions : 1 x nrows
    elif flag==2 :
        proj = np.dot(hor_filter,a)
        # Dimensions : 1 x ncols
    return proj

def get_base_texture(components , max_height , shape):
    constant_x = 0
    blocks =list()
    for each in components:
        cmy,cmx = tuple( [int(x) for x in ndimage.measurements.center_of_mass(each)] )
        block = np.ones((max_height , each.shape[1])) * 255
        block[max_height/2 - cmy : each.shape[0] - cmy + max_height/2 , :] = each
        blocks.append(block)
    for i in range(1,len(blocks)):
        blocks[0] = np.concatenate((blocks[0], blocks[i]), axis=1)
    return strip_white(blocks[0])

def get_blocks(base, shape):
    print ("Getting the blocks")
    blocks = list()
    lines = list()
    height , width = shape
    index = 0
    while index + width < base.shape[1]:
        lines.append(base[:, index : index+width ])
        index += width
    lines = [strip_white(x) for x in lines]
    block = lines[0]
    for i in range(1,len(lines)):
        block = np.concatenate((block,lines[i]),axis = 0)
    index = 0
    while index + height < block.shape[0]:
        blocks.append(block[index : index+height, : ])
        index += height
    return blocks

def extract(name):
    # ids["test"] = -1
    writers = list()
    if name[-4:]==".png":          # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        print("Processing "+ name)
        img = cv2.imread("/home/chris/honours/IAM_hand/" + name, 0)
        b_img = 1 - cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = 255 - (b_img * (255 - img))
        components = get_connected_components(img)
        components = refine(components)
        baseTexture = get_base_texture(components , img.shape[0])
        blocks  = get_blocks(baseTexture , (300,300))
        count = 0
        print("Writing blocks")
        for block in blocks:
            x = cv2.imwrite("/home/chris/honours/IAM_block/" + name[0:-4] + '_' + str(count) + ".png" ,block)
            count += 1
    return writers

data_path = "/home/chris/honours/IAM_hand/"              # Path of the original dataset
output_path = "/home/chris/honours/IAM_block/"            # Path of the output folder
writers = open("/home/chris/honours/IAM_block/writerids.csv" , "w")
folderlist = os.listdir(data_path)
# folderlist.remove("writerids.csv")
folderlist.sort()

pool = multiprocessing.Pool(6)
result = pool.map(extract, folderlist)
