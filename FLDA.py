import numpy as np
from matplotlib import pyplot as plt
import pdb

filename = "data_fullimg_csv/allfeatures.csv"
outputfile = "allfeatures_LDA.csv"
train_file = open(filename,"r")
train_data=[]
train_class=[]
classbased = []

for line in train_file:
    l = line.strip()
    l = l.split(',')
    l = map(float , l)
    train_data.append(l[0:-1])
    train_class.append(int(l[-1]))

attribs = len(train_data[0])
temp = []
for i in range(min(train_class),max(train_class)+1):
    temp.append(0)
for i in range(len(train_class)):
    temp[train_class[i]] += 1;
pdb.set_trace()

for i in range(min(train_class),max(train_class)+1):
     classbased.append([])

overmean = np.zeros((attribs,1))

for i in range(len(train_data)):
    x=np.array(train_data[i]).reshape(attribs,1)
    classbased[train_class[i]].append(x)
    overmean = overmean + x
overmean = overmean / len(train_data)

lis = []
for i in range(len(classbased)):
    lis.append(len(classbased[i]))

means =[]
for i in range(len(classbased)):
    if len(classbased[i])>0:
        meanvector = classbased[i][0]
        for j in range(1,len(classbased[i])):
            meanvector = meanvector + classbased[i][j]
        meanvector = meanvector/len(classbased[i])
        means.append(meanvector)
    else:
        means.append(np.array(()))

finalscatter_i = np.zeros((attribs,attribs))
finalscatter_b = np.zeros((attribs,attribs))

for i in range(len(classbased)):
    if len(classbased[i])>0:
        y = means[i] - overmean
        y = np.dot(y,y.T)
        finalscatter_b = finalscatter_b + len(classbased[i])*y
        temp = (classbased[i][0]-means[i])
        scattermat_i = np.dot(temp,temp.T)
        for j in range(len(classbased[i])):
            temp = (classbased[i][0]-means[i])
            scattermat_i = scattermat_i + np.dot(temp,temp.T)
    finalscatter_i = finalscatter_i + scattermat_i

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(finalscatter_i).dot(finalscatter_b))
pdb.set_trace()
