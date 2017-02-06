import pdb
from sklearn.lda import LDA

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = int(line[1])
    return dictionary

names1 = ["GSCM/GSCM_1","GSCM/GSCM_2","GSCM/GSCM_3","GSCM/GSCM_4","GSCM/GSCM_5","GSCM/GSCM_all"]
names2 = ["Gabor/Gabor_all", "Gabor/Gabor_4", "Gabor/Gabor_8", "Gabor/Gabor_16", "Gabor/Gabor_32", ]
names3 = ["Edge/Edge_all","Edge/Edge_8","Edge/Edge_12","Edge/Edge_16","Edge/Edge_dp_16",]
names4 = ["LBP/LBP"]
names = ["LBP3","LBP4","LBP5","LBP6","LBP7",]
for filename in names:
    path = "/home/chris/honours/Texture_Analysis/source/featurex/"

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
    ids = get_ids("/home/chris/writerids.csv")
    tr_class = [ids[x.split('-')[0]+'-'+x.split('-')[1] + '.png'] for x in train_class]
    clf = LDA()
    trans = clf.fit_transform(train_data,tr_class)
    f = open(path+filename+"_LDA.csv",'w')
    for i in range(len(train_data)):
        for entry in trans[i]:
            f.write(str(entry)+',')
        f.write(str(train_class[i])+'\n')
    f.close()
