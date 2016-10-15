import cv2
import numpy as np
import pdb
import random
import os
import skimage.feature
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

def normalize_img2(img,xml):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    root=xml.getroot()
    handwritten_text = root.findall("handwritten-part")[0]
    lines = handwritten_text.findall("line")
    line_list = []
    for line in lines:
        words = line.findall("word")
        cords = []
        max_height_word = 0
        for word in words:
            characters = word.findall("cmp")
            if len(characters)>0:

                max_height = int(characters[0].attrib["height"])
                min_cord = int(characters[0].attrib["y"])
                max_cord  = max_height + min_cord
                for c in characters:
                    length = int(c.attrib["height"])
                    y = int(c.attrib["y"])
                    length = length + y
                    if length > max_cord:
                        max_cord = length
                    if min_cord > y:
                        min_cord = y
                if len(characters)==1:
                    start_row = int(characters[0].attrib["y"])
                    end_row = start_row + int(characters[0].attrib["height"])
                    start_col = int(characters[0].attrib["x"])
                    end_col = start_col + int(characters[0].attrib["width"])
                else:
                    start_row = min_cord
                    end_row = max_cord
                    start_col = int(characters[0].attrib["x"])
                    end_col = int(characters[-1].attrib["x"]) + int(characters[-1].attrib["width"])
                cords.append([start_row,end_row,start_col,end_col])
                if (max_height_word < end_row - start_row):
                    max_height_word = end_row - start_row
        for cord in cords:
            if cord[1]-cord[0]!=max_height_word:
                cord[1]= max_height_word + cord[0]

        new_line = img[cords[0][0]:cords[0][1],cords[0][2]:cords[0][3]]
        for l in range(1,len(cords)):
            new_line = np.concatenate((new_line,img[cords[l][0]:cords[l][1],cords[l][2]:cords[l][3]]),1)
        line_list.append(new_line)
    max_line_length = line_list[0].shape[1]
    for line in line_list:
        x=line.shape[1]
        if x > max_line_length:
            max_line_length = x
    for line in range(len(line_list)):
        if line_list[line].shape[1]<max_line_length:
            line_list[line] = np.concatenate((line_list[line],np.ones((line_list[line].shape[0],max_line_length-line_list[line].shape[1]))),1)
    finalimage = line_list[0]
    for line in range(1,len(line_list)):
        finalimage = np.concatenate((finalimage,line_list[line]),0)
    return finalimage

def get_ids():
    f = open("writerids","r")
    dictionary = {}
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

folder = "/home/chris/honours/Texture_Analysis/fullimg/"
outfolder = "/home/chris/honours/Texture_Analysis/fullimg_norm/"
folderlist = os.listdir(folder)
f = open(outfolder+"writerids.csv","w")
print "Starting loop"
for name in folderlist:
    if name!="xml":
        n = name[0:-4]
        print "Processing "+ n + ".png"
        try:
            tree = ET.parse(folder+"xml/"+n+'.xml')
            root = tree.getroot()
            a=root.attrib
            label = a['writer-id']
            img = cv2.imread(folder+n+".png")
            img = normalize_img2(img,tree)
            img = img*255
            f.write(n+','+label+'\n')
            x = cv2.imwrite(outfolder+n+".png",img)
            print x
        except IOError:
            print "Error at "+ name
