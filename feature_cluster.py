from paths import *

import csv
import pdb
from sklearn.cluster import KMeans
import numpy as np

def getIds(filename):
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f , delimiter=',')
        for row in reader:
            ids[ row[0][0:-4] ] = int( row[1] )
    return ids


data = list()
ids = getIds(BASE_DIR+"writerids.csv")
results = ()
with open(BASE_DIR+"distribution.csv", 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        data.append([int(each) for each in row[0:-1]])
        data[-1].append(ids[row[-1]])
    data = np.array(data)
    for n in range(8, 9):
        kmeans = KMeans(n_clusters= n, init='k-means++', ).fit(data[:,0:-1])
        count = 0
        for i in range(1,104):
            indx = (data[:,-1]==i).nonzero()[0].tolist()
            labels = np.array(kmeans.labels_)[indx]
            if (labels == labels[0]).all():
                count+=1
            else:
                print i, labels 
        print n, count
