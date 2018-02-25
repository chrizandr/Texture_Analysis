"""SVM Classification."""

import numpy as np
from sklearn.svm import SVC
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


def classify(filename, top=1):
    """Classification."""
    ids = getIds("/home/chrizandr/data/firemaker/writerids.csv")
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
        svm = SVC(kernel='linear', probability=False, decision_function_shape='ovo')
        # svm = KNeighborsClassifier()
        svm.fit(t_train[:, 0:-1], t_train[:, -1])
        svm.decision_function_shape = 'ovr'
        result = svm.decision_function(t_test[:, 0:-1])
        sresult = np.argsort(result, axis=1)
        fresult = sresult[:, -top:]
        correct = 0
        # pdb.set_trace()
        for i in range(result.shape[0]):
            if int(t_test[i, -1]) in fresult[i]:
                correct += 1
        results.append((100*float(correct)) / t_test.shape[0])

        # result = svm.predict(t_test[:, 0:-1])
        # correct = 0
        # for i in range(result.shape[0]):
        #     if int(t_test[i, -1]) == result[i]:
        #         correct += 1
        # results.append((100*float(correct)) / t_test.shape[0])
    return sum(results)/len(results)


if __name__ == "__main__":
    results = list()
    for data_dir in ["/home/chrizandr/Texture_Analysis/source/strokes/"]:
        for filename in ["strokes_auto_29dutch"]:
            print("Classifying : " + filename)
            for i in range(1, 11):
                evl = classify(data_dir+filename+".csv", top=i)
                print("Evaluating")
                # ------------------------------------
                results.append((filename, evl))
    pdb.set_trace()
