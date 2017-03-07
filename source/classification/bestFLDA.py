import pdb
from sklearn.lda import LDA
import matplotlib.pyplot as plt

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = int(line[1])
    return dictionary

for filename in ["Features/conv_234_4"]:
    path = "/home/chrizandr/Texture_Analysis/data_telugu_blocks/"

    f = open(path+filename+".csv",'r')
    train_data = list()
    train_class = list()
    for line in f:
        l = line.strip()
        l = l.split(',')
        ly = list(map(float , l[0:-1]))
        train_data.append(ly)
        train_class.append(l[-1])
    f.close()
    ids = get_ids("/home/chrizandr/data/writerids.csv")
    tr_class = [ids[x.split('-')[0]+'-'+x.split('-')[1] + '.png'] for x in train_class]
    results = list()
    for i in range(1,77):
        print i
        clf = LDA(store_covariance = True, n_components = i)
        trans = clf.fit_transform(train_data,tr_class)
        params = clf.get_params()
        f = open(path+filename+"_LDA1.csv",'w')
        for i in range(len(train_data)):
            for entry in trans[i]:
                f.write(str(entry)+',')
            f.write(str(train_class[i])+'\n')
        f.close()
        data_dir = "/home/chrizandr/Texture_Analysis/data_telugu_blocks/"
        name = "Features/conv_234_4_LDA1"
        evl = classify(data_dir + name + ".csv" , 1)
        results.append((i,evl))
    pdb.set_trace()
