import numpy as np
from sklearn.cluster import KMeans
import pdb
import scipy
import matplotlib.pyplot as plt
import cv2
from shutil import copyfile
import os

def cluster(X, n_clusters, file_name):
    kmeans = KMeans(n_clusters=n_clusters).fit(X)


    data_path="/home/sanny/honours/Texture_Analysis/output/"
    path="/home/sanny/honours/Texture_Analysis/reps/"
    #getting representative cluster name
    centers = kmeans.cluster_centers_
    for each in centers:
        each = each.reshape(1,100)
        dist = scipy.spatial.distance.cdist(X, np.array(each))
        import operator
        min_index, min_value = min(enumerate(dist), key=operator.itemgetter(1))
        print file_name[min_index]
        copyfile(data_path+file_name[min_index],path+file_name[min_index])

    

    cdata = {}
    cname = {}
    for i in range(len(kmeans.labels_)):
        key = kmeans.labels_[i]
        if key in cdata.keys():
           cdata[key].append(X[i])
           cname[key].append(file_name[i])
        else:
           cdata[key]=[np.array(X[i])]
           cname[key]=[file_name[i]]

    #plot each cluster images
    '''data_path="/home/sanny/honours/Texture_Analysis/output/"
    path = '/home/sanny/honours/Texture_Analysis/97clusters/'
    for cluster, path_list in cname.iteritems():
        cluster_path=path+str(cluster)+"/"
        os.makedirs(cluster_path)
        for each in path_list:
            copyfile(data_path+each,cluster_path+each)
            #i = cv2.imread(data_path+each, 0)
            #print cluster
            #plt.imshow(i, cmap='gray')
            #plt.show()'''
   
    #plot cluster center graph
    #plt.hist([[float(x) for x in y] for y in kmeans.cluster_centers_])
    #plt.show()
    #x=range(97)
    #centers = np.array(kmeans.cluster_centers_)
    #for i in range(len(centers)):
    #    plt.plot(x,[pt[i] for pt in centers], label = i)
    #plt.legend()
    #plt.show()

    t_var = 0
    for  cluster, data in cdata.iteritems():
          data = np.array(data)
          mean= data.mean(axis=0)
          var = (data - mean)**2
          t_var+= var.mean()
    return t_var/n_clusters



f = open('features.csv', 'r')
file_name =[]
fea = []
for line in f:
    line = line.strip().split(',')
    file_name.append( line[-1])
    fea.append([int(each) for each in line[:-1]])

X = np.array(fea)
lis = []
for i in range(97,98):
    v = cluster(X,i,file_name)
    #print i,"           ", v
    lis.append(v)

import operator
min_index, min_value = min(enumerate(lis), key=operator.itemgetter(1))
print min_index, min_value
plt.plot(range(97,98),lis)
plt.xlabel("No. of clusters")
plt.ylabel("Avg within-cluster variance")
plt.show()


