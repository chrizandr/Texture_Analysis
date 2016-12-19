import cv2
import numpy as np
import pdb
import random
import os
import skimage.feature
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
#CODE FOR NORMALIZATION OF THE TEXT IMAGE
#---------------------------------------------------------------------------------------------------
def segment_text(img):
    #convert image to Greyscale
    a = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #convert image to binary using OTSU thresholding algorithm
    ret, a = cv2.threshold(a , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Get the text part of page
    central_cols = a[:,900:1300]
    ver_proj_cl = projection(central_cols,1)
    threshold = 50
    indices_a = (ver_proj_cl<threshold).nonzero()
    index=[]
    for i in range(1,len(indices_a[0])):
        if ((indices_a[0][i]-indices_a[0][i-1])>100):
            index.append(indices_a[0][i])
    if len(index)==2:
        start_row=index[0]+50
        end_row=index[1]-50
    a_textblck = a[start_row:end_row,:]
    #vertically segment actual text
    ver_proj = projection(a_textblck,1)
    threshold = a_textblck.shape[1]
    indices = (ver_proj<threshold).nonzero()
    start_row=indices[0][0]
    end_row=indices[0][-1]
    #horizontally segment actual text
    hor_proj = projection(a_textblck,2)
    threshold = a_textblck.shape[0]
    indices = (hor_proj<threshold).nonzero()
    start_col=indices[1][0]
    end_col=indices[1][-1]
    handwritten_text = a_textblck[start_row:end_row,start_col:end_col]
    return handwritten_text

def projection(a,flag):
    nrows = a.shape[0] #m
    ncols = a.shape[1] #n
    print nrows,ncols
    #creating filters for projection based on dimensions
    ver_filter = np.ones((ncols,1))
    hor_filter = np.ones((1,nrows))
    #vertical projection calculated
    if flag==1:
        proj = np.dot(a,ver_filter)
    elif flag==2:
        proj = np.dot(hor_filter,a)
    return proj

def normalize_img(img):
    ver_proj=projection(img,1)
    threshold = img.shape[1]
    start_row = 0
    finalimage=np.array(())
    for i in range(1,img.shape[0]):
        if(ver_proj[i]==threshold or i==img.shape[0]-1):
            end_row = i-1
            if i==img.shape[0]:
                break
            if finalimage.shape[0]==0:
                finalimage = normalize_line(img[start_row:end_row,:])
            else:
                finalimage = np.concatenate((finalimage,img[start_row:end_row,:]),0)
            while(ver_proj[i]==threshold):
                i=i+1
            start_row = i
    finalimage = np.concatenate((finalimage,img[start_row:end_row,:]),0)
    return finalimage

def normalize_line(img):
    hor_proj=projection(img,2)
    threshold = img.shape[0]
    finalimage=np.array(())
    flag = False
    for i in range(0,img.shape[1]):
        if(flag == False and hor_proj[0][i] == threshold):
            start_col = i
            flag = True
        if(flag == True and hor_proj[0][i]!=threshold):
            end_col = i-1
            if start_col==0:
                finalimage = np.concatenate((np.ones((img.shape[0],1)),img[:,end_col+1:]),1)
            else:
                finalimage = np.concatenate((finalimage[:,:start_col+1],finalimage[:,end_col:]),1)
            flag = False
    finalimage = np.concatenate((finalimage,np.ones((img.shape[0],img.shape[1]-finalimage.shape[1]))),1)
    return finalimage

#CODE FOR FEATURE SET A : GABOR FEATURES
#---------------------------------------------------------------------------------------------------
def feature_setA(img,frequencies,angles,sigma):
    features=[]
    for f in frequencies:
        for theta in angles:
            x=get_Gabor_features(img,f,theta,sigma)
            a=x.mean()*100
            b=x.std()*100
            features.append(a)
            features.append(b)
    return features

def get_Gabor_features(img,f,theta,sigma):
    theta = getangle(theta)
    gaussian_kernel = make_gaussian_kernel((100,100),sigma)
    Gka, Gkb = make_Gabor_kernel(gaussian_kernel,f,theta)
    pdb.set_trace()
    img1 = cv2.filter2D(img,-1,Gka)
    img2 = cv2.filter2D(img,-1,Gkb)
    img1 = img1*img1
    img2 = img2*img2
    x = img1+img2
    feature = np.sqrt(x)
    return feature

def make_Gabor_kernel(gkernel,f,theta):
    he=np.arange(gkernel.size).reshape(gkernel.shape)
    ho=np.arange(gkernel.size).reshape(gkernel.shape)
    he=he.astype(np.float64)
    ho=ho.astype(np.float64)
    for i in range(he.shape[0]):
        for j in range(he.shape[1]):
            val= 2*np.pi*f*((i+1)*np.cos(theta)+(j+1)*np.sin(theta))
            he[i,j]=gkernel[i,j]*np.cos(val)

    for i in range(ho.shape[0]):
        for j in range(ho.shape[1]):
            val= 2*np.pi*f*((i+1)*np.cos(theta)+(j+1)*np.sin(theta))
            ho[i,j]=gkernel[i,j]*np.sin(val)
    return he,ho

def make_gaussian_kernel(ksize, sigma):
    kx=cv2.getGaussianKernel(ksize[0],sigma);
    ky=cv2.getGaussianKernel(ksize[1],sigma);
    kernel = kx*ky.transpose()
    return kernel

def getangle(n):
    return (np.float64)(n*np.pi)/180

#CODE FOR FEATURE SET B : GSCM FEATURES
#---------------------------------------------------------------------------------------------------
def feature_setB(img,distances,angles):
    for i in range(len(angles)):
        angles[i]=getangle(angles[i])
    features=[]
    x = skimage.feature.greycomatrix(img,distances,angles,2)
    #pdb.set_trace()
    for d in range(len(distances)):
        for theta in range(len(angles)):
            features.append(x[0,0,d,theta])
            features.append(x[1,0,d,theta])
            features.append(x[1,1,d,theta])
    return features

def normalize_img2(img,xml):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    root=xml.getroot()
    handwritten_text = root.findall("handwritten-part")[0]
    lines = handwritten_text.findall("line")
    line_list = []
    count_line = 0
    for line in lines:
        print "line:",count_line
        count_line+=1
        words = line.findall("word")
        cords = []
        max_height_word = 0
        countword = 0
        for word in words:
            print "word:",countword
            countword+=1
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
            new_line = np.concatenate((new_line,np.ones((max_height_word,5)),img[cords[l][0]:cords[l][1],cords[l][2]:cords[l][3]]),1)
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

img = cv2.imread("/home/chris/honours/Project_INDICON/raw_data/SET_1/a01-053u.png")
metadata=ET.parse("/home/chris/xml/a01-053u.xml")
img=normalize_img2(img,metadata)
img = img[100:3200,200:2400]
img = segment_text(img)
img = normalize_img(img)
sigma = 2
frequencies = [4,8,16,32]
angles = [0,45,90,135]
distances = [1,2,3,4,5]
A1 = feature_setA(img,frequencies,angles,sigma)
A2 = feature_setB(img,distances,angles)
