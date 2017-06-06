import numpy as np
from sklearn.cluster import KMeans
import pdb
import scipy
import matplotlib.pyplot as plt
import cv2
from shutil import copyfile
import os
import operator


def cluster(X, n_clusters, file_name):
    """."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', ).fit(X)
    output = kmeans.labels_

    # data_path = "/home/sanny/honours/Texture_Analysis/output/"
    # path = "/home/sanny/honours/Texture_Analysis/reps/"
    # # getting representative cluster name
    # centers = kmeans.cluster_centers_
    # for each in centers:
    #     each = each.reshape(1, 100)
    #     dist = scipy.spatial.distance.cdist(X, np.array(each))
    #     min_index, min_value = min(enumerate(dist), key=operator.itemgetter(1))
    #     print file_name[min_index]
    #     copyfile(data_path+file_name[min_index], path+file_name[min_index])

    #pdb.set_trace()
    #data_path="/home/sanny/honours/Texture_Analysis/output/"
    #path = '/home/sanny/honours/Texture_Analysis/clusters/'
    cdata = {}
    cname = {}
    for key in set(output):
        indices = (output == key).nonzero()[0]
        points = X[indices]
        cdata[key] = points
    #    cname[key] = indices
    #    cluster_path = path+str(key)+"/"
    #    os.makedirs(cluster_path)
    #    for f in indices:
    #        each = file_name[f]
    #        copyfile(data_path+each, cluster_path+each)

    # plot cluster center graph
    # plt.hist([[float(x) for x in y] for y in kmeans.cluster_centers_])
    # plt.show()
    # x=range(97)
    # centers = np.array(kmeans.cluster_centers_)
    # for i in range(len(centers)):
    #    plt.plot(x,[pt[i] for pt in centers], label = i)
    # plt.legend()
    # plt.show()

    t_var = 0
    for cluster, data in cdata.iteritems():
         data = np.array(data)
         mean = data.mean(axis=0)
         var = (data - mean)**2
         t_var += var.mean(axis=0).sum()
    return t_var/n_clusters


print("Getting from file...")
f = open('features.csv', 'r')
file_name = []
fea = []
for line in f:
    line = line.strip().split(',')
    file_name.append(line[-1])
    fea.append([float(each) for each in line[:-1]])

print("Converting to array")
X = np.array(fea)
lis = []
for i in range(97, 98):
    print("Custering for k = ", i)
    v = cluster(X, i, file_name)
    print(i, v)
    lis.append(v)

min_index, min_value = min(enumerate(lis), key=operator.itemgetter(1))
print min_index, min_value
plt.plot(range(97, 98), lis)
plt.xlabel("No. of clusters")
plt.ylabel("Avg within-cluster variance")
plt.show()
