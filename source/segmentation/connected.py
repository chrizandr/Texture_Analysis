import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import multiprocessing

def find_position(flag , shape):
    for i in range(flag.shape[0] - shape[0]):
        for j in range(flag.shape[1] - shape[1]):
            if flag[i,j] != 1:
                if 1 not in flag[ i : i+shape[0] , j : j+shape[1] ]:
                    return (i,j)
    return -1

def refine(components):
    useful = list()
    for component in components:
        comp = cv2.threshold(component , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        struct = np.ones((3,3),dtype=np.uint8)
        img = cv2.dilate(1-comp,struct,iterations=1)
    	img = cv2.erode(img,struct,iterations=1)
    	high = (img.shape[0]*img.shape[1])*0.75
    	low = (img.shape[0]*img.shape[1])*0.25
    	isum = (1-img).sum()
    	if isum > low and isum < high:
        	useful.append(component)
    return useful

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

def get_text_blocks(components):
    blocks = list()
    # flag = np.zeros((300,300))
    block = np.ones((300,300)) * 255
    x = 0
    y = 0
    print("Getting blocks")
    for component in components:
        # position = find_position(flag, component.shape)
        if component.shape[0] > 300 or component.shape[1] > 300:
            print "block left"
            continue
        if x + component.shape[1] < 300 and y + component.shape[0] < 300:
            block[y:y+component.shape[0]]


        #
        # if position == -1:
        #     blocks.append(block)
        #     block = np.ones((300,300)) * 255
        #     flag = np.zeros((300,300))
        #     position = (0,0)
        # block[position[0]:position[0]+component.shape[0] , position[1]:position[1]+component.shape[1]] = component
        # flag[position[0]:position[0]+component.shape[0] , position[1]:position[1]+component.shape[1]] = 1
    return blocks

def get_ids():
    f = open("/home/chris/honours/Texture_Analysis/writerids.csv","r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

def extract(name):
    ids = get_ids()
    if name[-4:]==".png":          # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        print("Processing "+ name)
        label = ids[name[0:-4]]
        img = cv2.imread("/home/chris/honours/hand_img/" + name , 0)
        components = get_connected_components(img)
        components = refine(components)
        for component in components:
            plt.imshow(component,'gray')
            plt.show()
        blocks = get_text_blocks(components)
        count = 0
        print("Writing blocks")
        for block in blocks:
            x = cv2.imwrite("/home/chris/honours/text_blocks/" + name[0:-4] + str(count) + ".png" ,block * 255)
            writers.write(name[0:-4]+str(count)+','+ label + '\n')
            count += 1
    return None


# Optional main function
data_path = "/home/chris/honours/hand_img/"              # Path of the original dataset
output_path = "/home/chris/honours/text_blocks/"            # Path of the output folder
writers = open("/home/chris/honours/text_blocks/writerids.csv" , "w")
# Get a list of all the files in the dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.sort()
# Open the output file in write mode
print("Starting........")
img = cv2.imread("test.png",0)
components = get_connected_components(img)
components = refine(components)
block = get_text_blocks(components)
pdb.set_trace()

for component in components:
    plt.imshow(component,'gray')
    plt.show()

#
# pool = multiprocessing.Pool(4)
# pool.map(extract, folderlist)
