import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage.filters import sobel,threshold_otsu
from skimage import io
import time
import itertools

def edge_detection(img):
    img=sobel(img)
    thresh=threshold_otsu(img)
    img[img>thresh] = 1
    img[img<=thresh] = 0
    img=img.astype(int)
    return img

def Hinge_detection(img):
    (m,n) = img.shape
    orient16 = np.zeros((16,16))
    orient24 = np.zeros((24,24))
    orient32 = np.zeros((32,32))
    all_orient=[0,orient16,orient24,orient32]
    orient_map=dict()
    for i in range(4,m-4):
        for j in range(4,m-4):
            for x in range(1,5):
                l=[0 for i in range(0,32)]
                orient_map=dict()
                layer_index = [(i+x,j+x)]
                count=0
                orient_map[(i+x,j+x)]=count
                for k in range(1,((2*x)+1)):
                    layer_index.append((i+x,j+x-k))
                    orient_map[(i+x,j+x-k)]=count
                    count+=1
                for k in range(1,2*x+1):
                    layer_index.append((i+x-k,j-x))
                    orient_map[(i+x-k,j-x)]=count
                    count+=1
                for k in range(1,2*x+1):
                    layer_index.append((i-x,j-x+k))
                    orient_map[(i-x,j-x+k)]=count
                    count+=1
                for k in range(1,2*x):
                    layer_index.append((i-x+k,j+x))
                    orient_map[(i-x+k,j+x)]=count
                    count+=1
                #print orient_map
                for t in layer_index:
                    a=t[0]
                    b=t[1]
                    if img[i][j]==1 and img[a,b]==1:
                        if abs((a-i))==abs((b-j)):
                            if img[a-1][b-1]==1:
                                l[orient_map[t]]+=1
                            else:
                                img[a,b]=0
                        else:
                            if (j+x)==b:
                                if abs((a+1-i))==abs((b-j)):
                                    if (img[a][b-1]==1 or img[a-1][b-1]==1):
                                        l[orient_map[t]]+=1
                                    else:
                                        img[a,b]=0
                                elif img[a][b-1]==1 or img[a+1][b-1]==1 or img[a-1][b-1]==1:
                                    l[orient_map[t]]+=1
                                else:
                                    img[a][b]=0
                            elif (j-x)==b:
                                if abs((a+1-i))==abs((b-j)):
                                    if (img[a][b+1]==1 or img[a-1][b+1]==1):
                                        l[orient_map[t]]+=1
                                    else:
                                        img[a,b]=0
                                elif img[a][b+1]==1 or img[a+1][b+1]==1 or img[a-1][b+1]==1:
                                    l[orient_map[t]]+=1
                                else:
                                    img[a][b]=0
                            elif (i+x)==a:
                                if abs((a-i))==abs((b-j-1)):
                                    if (img[a-1][b]==1 or img[a-1][b-1]==1):
                                        l[orient_map[t]]+=1
                                    else:
                                        img[a,b]=0
                                elif img[a-1][b]==1 or img[a-1][b+1]==1 or img[a-1][b-1]==1:
                                    l[orient_map[t]]+=1
                                else:
                                    img[a][b]=0
                            elif (i-x)==a:
                                if abs((a-i))==abs((b-j-1)):
                                    if (img[a+1][b]==1 or img[a+1][b-1]==1):
                                        l[orient_map[t]]+=1
                                    else:
                                        img[a,b]=0
                                elif img[a+1][b]==1 or img[a+1][b+1]==1 or img[a+1][b-1]==1:
                                    l[orient_map[t]]+=1
                                else:
                                    img[a][b]=0
                            #else:
                            #    print i,j,'------',t,"unsuccssful"
                    if (x-1)!=0:
                        for m in range(0,32):
                            for n in range(0,32):
                                if l[m]==1 and l[n]==1:
                                    all_orient[x-1][m][n]+=1
    return all_orient

start_time = time.time()
img = io.imread('trial.png')
img = edge_detection(img)
features = Hinge_detection(img)

print("--- %s seconds ---" % (time.time() - start_time))
