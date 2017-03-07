import csv, pdb

def getIds(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        l = line.strip()
        l = l.split(",")
        dictionary[l[0]] = l[1]
    return dictionary

reader = csv.reader( open("after.csv" , "rb"), delimiter=',')
data = list()
classes = list()
for row in reader:
    data.append(row[0:-1])
    classes.append(row[-1])
ids = getIds("/home/chrizandr/data/writerids.csv")
classes = [c.split('-')[0] + '-' + c.split('-')[1] + '.png' for c in classes]
classes = [ids[c] for c in classes]
f = open("after1.csv", "wb")
for i in range(len(data)):
    for d in data[i]:
        f.write(d+'\t')
    f.write(classes[i]+'\n')
f.close()
pdb.set_trace()
