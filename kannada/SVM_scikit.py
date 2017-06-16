import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pdb,csv
import matplotlib.pyplot as plt
from collections import Counter

def plt_dist(t_train, id):
    writer = t_train[(t_train[:,-1]==id).nonzero(),:-1][0]
    a = writer[0]/writer[0].sum()
    b = writer[1]/writer[1].sum()
    c = writer[2]/writer[2].sum()
    plt.plot( a )
    plt.plot( b )
    plt.plot( c )
    plt.show()


def classify(filename):
    f = open(filename , 'r')
    reader = csv.reader(f , delimiter=',')
    dataset = list()
    filename = list()
    for row in reader:
        data = [float(r) for r in row[0:-1]]
        writer_id = int(row[-1].split('_')[2].split('-')[0])
        data.append(writer_id)
        dataset.append(data)
        filename.append(row[-1].split('-')[0])
    
    dataset = np.array(dataset)
    results = list()


    train = list()
    test = list()
    test_files = list()
    train_files = list()
    for i in range(dataset.shape[0]):
        if filename[i].startswith('Kannada_3'):
            test.append(dataset[i,:])
            test_files.append(filename[i])
        else:
            train.append(dataset[i,:])
            train_files.append(filename[i])
    t_train = np.array(train)
    t_test = np.array(test)
    
    ### Document-wise classification
    #for i in range(len(train_files)):
    
    k = 16    #to change number of lines while adding

    ## 55 - 62 : to get indices of all lines that belong to a page
    names = list(Counter(train_files))
    t_ind = list()
    for name in names:
        lis = list()
        for i in range(len(train_files)):
            if train_files[i] == name and len(lis)<k:
                lis.append(i)
        t_ind.append(lis)    

    # 65 - 70 : use indices and add line distributions to make 171 ka array containing train data
    tr_data = list()
    for each in t_ind:
        x = t_train[np.array(each),:-1]
        x = x.sum(axis=0).tolist()
        x.append(t_train[each[0],-1])
        tr_data.append(x)


    # similar for test data
    names = list(Counter(test_files))
    t_ind = list()
    for name in names:
        lis = list()
        for i in range(len(test_files)):
            if test_files[i] == name and len(lis)<k:
                lis.append(i)
        t_ind.append(lis)
    
    ts_data = list()
    for each in t_ind:
        x = t_test[np.array(each),:-1]
        x = x.sum(axis=0).tolist()
        x.append(t_test[each[0],-1])
        ts_data.append(x)

    t_train = np.array(tr_data)
    t_test = np.array(ts_data)
    print t_train.shape, t_test.shape
    ####

    svm = SVC(kernel='linear')
    # svm = KNeighborsClassifier()
    svm.fit(t_train[:,0:-1],t_train[:,-1])
    result = svm.predict(t_test[:,0:-1])
       
    correct =  np.count_nonzero(np.equal(result, t_test[:,-1]))

    return correct/float(result.shape[0])

results = list()
for data_dir in ["/home/sanny/honours/k_experiments/"]:
    for filename in ["distributions"]:
        print("Classifying : " + filename)
        evl = classify(data_dir+filename+".csv")
        print(evl)
        # ------------------------------------
        results.append((filename, evl))
