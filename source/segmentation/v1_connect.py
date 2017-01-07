import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import multiprocessing
from scipy import ndimage
import math

def projection(a,flag):
    # ------------------------------------
    # Projecting the image on the horizontal or vertical axis by multiplying with vector of 1's
    nrows,ncols = a.shape
    ver_filter = np.ones((ncols,1))
    hor_filter = np.ones((1,nrows))
    # ----------------------
    # vertical projection, the array needs to be transposed to make it into a vector
    if flag==1 :
        proj = np.dot(a,ver_filter)
        proj = proj.T
        # Dimensions : 1 x nrows
    # ----------------------
    # horizontal projection, the array is already in the for of a vector
    elif flag==2 :
        proj = np.dot(hor_filter,a)
        # Dimensions : 1 x ncols
    # ------------------------------------
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
    print("Getting the strong components")
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
    print("Getting the base texture")
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
    blocks = list()
    texture_height, texture_width = base.shape
    lines = list()
    index = 0
    while index + width < texture_width:
        lines.append(texture[:, index : index+width ])
        index += width
    block=np.ones(shape) * 255
    k = 0
    y = int(texture_height/2)
    for line in lines:
        if k+texture_height < shape[0]:
            block[k:k+texture_height,:]=np.logical_and(line,block[k:k+texture_height,:])
            k+=y
        else:
            blocks.append(block)
            k = 0
    for each in blocks:
        plt.imshow(each,'gray')
        plt.show()
    return blocks

img = cv2.imread("test.png", 0)
# img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
components = get_connected_components(img)
components = refine(components)
baseTexture = get_base_texture(components , img.shape[0])
blocks  = get_blocks(baseTexture , (300,300))








'''
for each in components:
    m,n = each.shape
    new_img = np.zeros((m,n,3))
    new_img[:,:,0] = each.copy()
    new_img[:,:,1] = each.copy()
    new_img[:,:,2] = each.copy()
    cm = ndimage.measurements.center_of_mass(each)
    cmx = math.floor(cm[0])
    cmy = math.floor(cm[1])
    print cmx,cmy
    new_img[cmx,cmy,0] = 12
    new_img[cmx,cmy,1] = 100
    new_img[cmx,cmy,2] = 100
    print m-cmy
    plt.imshow(new_img)
    plt.show()
'''
