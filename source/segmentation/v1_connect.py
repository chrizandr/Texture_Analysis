import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import multiprocessing
from scipy import ndimage
import math

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

def strip_white(img):
    m,n = img.shape
    flag = 1
    proj = projection(img,1)
    cords = (proj!=255*n).nonzero()
    return img[cords[1][0]:cords[1][-1]+1 , :]

def get_connected_components(img):
    b_img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    nimg = cv2.connectedComponents(1-b_img)[1]
    print("Getting the connectedComponents")
    labels = set(np.unique(nimg))
    labels.remove(0)
    components = list()
    for label in labels:
        sub_region = (nimg==label).nonzero()
        max_hor = sub_region[1].max()
        min_hor = sub_region[1].min()
        max_ver = sub_region[0].max()
        min_ver = sub_region[0].min()
        region = img[min_ver : max_ver, min_hor :max_hor]
        if max_hor - min_hor > 3 and max_ver - min_ver > 3:
            components.append(region)
    return components

def refine(components):
    useful = list()
    for component in components:
        comp = cv2.threshold(component , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        struct = np.ones((3,3),dtype=np.uint8)
        img = cv2.dilate(1-comp,struct,iterations=1)
    	img = cv2.erode(img,struct,iterations=1)
    	high = (img.shape[0]*img.shape[1])*0.75
    	low = (img.shape[0]*img.shape[1])*0.25
    	isum = (img).sum()
    	if isum < high:
        	useful.append(component)
    return useful

def get_base_texture(components , max_height):
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

data_path = "/home/sanny/honours/size_test/"              # Path of the original dataset
output_path = "/home/chris/honours/block_300_300/"            # Path of the output folder
writers = open("/home/chris/honours/writerids300300.csv" , "w")
folderlist = os.listdir(data_path)
# folderlist.remove("writerids.csv")
folderlist.sort()

pool = multiprocessing.Pool(5)
result = pool.map(extract, folderlist)
