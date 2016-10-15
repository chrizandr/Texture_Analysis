import numpy as np
import pdb
import time
import os
from skimage.filters import sobel,threshold_otsu
from skimage import io

def Edge_direction(binary):
    m,n=binary.shape
    orient8=[0,0,0,0,0,0,0,0,0]
    orient12=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    orient16=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(4,m):
      for j in range(4,n-4):
         if binary[i][j]==1:
             neigh=[[],[],[],[]]
             neigh[0]=[binary[i][j+1],binary[i-1][j+1],binary[i-1][j],binary[i-1][j-1],binary[i][j-1]]
             neigh[1]=[binary[i][j+2],binary[i-1][j+2],binary[i-2][j+2],binary[i-2][j+1],binary[i-2][j],binary[i-2][j-1],binary[i-2][j-2],binary[i-1][j-2],binary[i][j-2]]
             neigh[2]=[binary[i][j+3],binary[i-1][j+3],binary[i-2][j+3],binary[i-3][j+3],binary[i-3][j+2],binary[i-3][j+1],binary[i-3][j],binary[i-3][j-1],binary[i-3][j-2],binary[i-3][j-3],binary[i-2][j-3],binary[i-1][j-3],binary[i][j-3]]
             neigh[3]=[binary[i][j+4],binary[i-1][j+4],binary[i-2][j+4],binary[i-3][j+4],binary[i-4][j+4],binary[i-4][j+3],binary[i-4][j+2],binary[i-4][j+1],binary[i-4][j],binary[i-4][j-1],binary[i-4][j-2],binary[i-4][j-3],binary[i-4][j-4],binary[i-3][j-4],binary[i-2][j-4],binary[i-1][j-4],binary[i][j-4]]
             layer_height=2
             layer_halfwidth=1
             for p in range(1,4):
                 layer_height+=1
                 layer_halfwidth+=1
                 q=p-1;
                 x=0
                 y=1
                 z=2
                 for t in range(0,len(neigh[p])):
                     if neigh[p][t]==1 and (t==(layer_height-1)) or (t==(3*(layer_height-1))):
                         if neigh[q][y]!=1:
                            neigh[p][t]=0
                     elif neigh[p][t]==1:
                         if (t==(layer_height+layer_halfwidth-1)):
                            if (neigh[q][x]!=1 and neigh[q][y]!=1 and neigh[q][z]!=1):
                                neigh[p][t]=0
                            x+=1
                            y+=1
                            z+=1
                         elif t==0:
                            if (neigh[q][0]!=1 and neigh[q][1]!=1):
                             neigh[p][t]=0
                         elif t==(layer_height-2) or t==layer_height:
                             if neigh[q][y]!=1 and neigh[q][z]!=1:
                                 neigh[p][t]=0
                             if t==(layer_height-2):
                                 x+=1
                                 y+=1
                                 z+=1
                         elif t<(layer_height-2):
                             if neigh[q][x]!=1 and neigh[q][y]!=1 and neigh[q][z]!=1:
                                 neigh[p][t]=0
                             x+=1
                             y+=1
                             z+=1
                         elif t<((layer_halfwidth-1)*2+layer_height):
                              if neigh[q][x]!=1 and neigh[q][y]!=1 or neigh[q][z]!=1:
                                  neigh[p][t]=0
                              x+=1
                              y+=1
                              z+=1
                         elif ((t==(layer_halfwidth-1)*2+layer_height) or (t==(layer_halfwidth-1)*2+layer_height+2)):
                             if neigh[q][x]!=1 and neigh[q][y]!=1:
                                 neigh[p][t]==0
                             if (t==(layer_halfwidth-1)*2+layer_height-1):
                                x+=1
                                y+=1
                                z+=1
                         elif (t>(layer_halfwidth-1)*2+layer_height+2):
                             if t==(len(neigh[p])-1) and neigh[q][x]!=1 and neigh[q][y]!=1:
                                 neigh[p][t]=0
                             elif neigh[q][x]!=1 and neigh[q][y]!=1 and neigh[q][z]!=1:
                                 neigh[p][t]=0
                             x+=1
                             y+=1
                             z+=1
                     if neigh[p][t]==1:
                         if p==1 and t!=8 :
                             orient8[t]+=1
                             orient8[-1]+=1
                         elif p==2 and t!=12:
                              orient12[t]+=1
                              orient12[-1]+=1
                         elif p==3 and t!=16:
                              orient16[t]+=1
                              orient16[-1]+=1
    orient8=[round((i*100)/float(orient8[-1]),4) for i in orient8]
    orient12=[round((i*100)/float(orient12[-1]),4) for i in orient12]
    orient16=[round((i*100)/float(orient16[-1]),4) for i in orient16]

    return orient8[0:-1],orient12[0:-1],orient16[0:-1]

def get_ids(outfolder):
    f = open(outfolder+"writerids.csv","r")
    dictionary = {}
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

def differentiate(vect):
    new_vect =[0 for i in range(len(vect)-1)]
    for i in range(1,len(vect)):
        new_vect[i-1] = vect[i] - vect[i-1]
    return new_vect

outfolder = "/home/chris/honours/fullimg_norm/"
folderlist = os.listdir(outfolder)
dictionary = get_ids("")
folderlist.sort()
f = open(outfolder+"Edge_direction.csv","w")
print "Starting loop"
for name in folderlist:
    if len(name)!=13 and name!="writerids.csv" and name!="Edge_direction.csv":
        n = name[0:-4]
        print "Processing "+ n + ".png"
        start_time = time.time()
        img= io.imread(outfolder+n+".png");
        eimg=sobel(img)
        thresh=threshold_otsu(eimg)
        eimg[eimg>thresh] = 1
        eimg[eimg<=thresh] = 0
        binary=eimg
        binary=binary.astype(int)
        orient8,orient12,orient16 = Edge_direction(binary)
        diff_orient16 = differentiate(orient16)
        for i in orient8:
            f.write(str(i)+',')
        for i in orient12:
            f.write(str(i)+',')
        for i in orient16:
            f.write(str(i)+',')
        for i in diff_orient16:
            f.write(str(i)+',')
        label = dictionary[n]
        f.write(label)
        print("--- %s seconds ---" % (time.time() - start_time))
print "Done :)"
f.close()
