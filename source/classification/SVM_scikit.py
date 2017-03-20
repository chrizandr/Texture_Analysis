import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pdb,csv

def getIds(filename):
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f , delimiter=',')
        for row in reader:
            ids[ row[0][0:-4] ] = int( row[1] )
    return ids

def classify(filename):
    ids = getIds("/home/chrizandr/data/writerids.csv")
    f = open(filename , 'r')
    reader = csv.reader(f , delimiter=',')
    dataset = list()
    for row in reader:
        data = [float(r) for r in row[0:-1]]
        data.append(ids[row[-1]])
        dataset.append(data)
    dataset = np.array(dataset)
    results = list()
    for model in range(10):
        np.random.shuffle(dataset)
        labels = dataset[:,-1:]
        classes = [int(x) for x in np.unique(labels)]
        train = list()
        test = list()
        for i in range(dataset.shape[0]):
            if int(dataset[i,-1]) in classes:
                test.append(dataset[i,:])
                classes.remove(dataset[i,-1])
            else:
                train.append(dataset[i,:])
        t_train = np.array(train)
        t_test = np.array(test)
        svm = SVC(kernel='linear')
        # svm = KNeighborsClassifier()
        svm.fit(t_train[:,0:-1],t_train[:,-1])
        result = svm.predict(t_test[:,0:-1])
        correct = 0
        for i in range(result.shape[0]):
            if int(result[i]) == int(t_test[i,-1]):
                correct +=1
        results.append((100*float(correct)) / t_test.shape[0])
    return sum(results)/len(results)

results = list()
for data_dir in ["/home/chrizandr/Texture_Analysis/noise/"]:
    for filename in ["Edge/Edge_ng_1","Edge/Edge_ng_2","Edge/Edge_ng_3","Edge/Edge_ng_4","Edge/Edge_ng_5",]:
        print("Classifying : " + filename)
        evl = classify(data_dir+filename+".csv")
        print("Evaluating")
        # ------------------------------------
        results.append((filename.split('/')[1], evl))
pdb.set_trace()
