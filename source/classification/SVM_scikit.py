"""SVM Classification."""

import numpy as np
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
import pdb
import csv


def getIds(filename):
    """Get writer ids."""
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ids[row[0][0:-4]] = int(row[1])
    return ids


def classify(filename):
    """Classification."""
    ids = getIds("/home/chrizandr/data/Telugu/writerids.csv")
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter=',')
    dataset = list()
    for row in reader:
        data = [float(r) for r in row[0:-1]]
        data.append(ids[row[-1]])
        dataset.append(data)

    dataset = np.array(dataset)
    results = list()
    for model in range(10):
        np.random.shuffle(dataset)
        labels = dataset[:, -1:]
        classes = [int(x) for x in np.unique(labels)]
        train = list()
        test = list()
        for i in range(dataset.shape[0]):
            if int(dataset[i, -1]) in classes:
                test.append(dataset[i, :])
                classes.remove(dataset[i, -1])
            else:
                train.append(dataset[i, :])
        t_train = np.array(train)
        t_test = np.array(test)
        svm = SVC(kernel='linear')
        # svm = KNeighborsClassifier()
        svm.fit(t_train[:, 0:-1], t_train[:, -1])
        result = svm.predict(t_test[:, 0:-1])
        correct = 0
        for i in range(result.shape[0]):
            if int(result[i]) == int(t_test[i, -1]):
                correct += 1
        results.append((100*float(correct)) / t_test.shape[0])
    return sum(results)/len(results)


if __name__ == "__main__":
    results = list()
    for data_dir in ["/home/chrizandr/Texture_Analysis/source/strokes/"]:
        for filename in ["distributions"]:
            print("Classifying : " + filename)
            evl = classify(data_dir+filename+".csv")
            print("Evaluating")
            # ------------------------------------
            results.append((filename, evl))
    pdb.set_trace()
