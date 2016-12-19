import pdb

featureA = open("data_fullimg_csv/GSCM_fullimg/GSCM_all.csv","r")
featureB = open("data_fullimg_csv/Edge_direction_fullimg/Edge_all.csv","r")
feature = open("data_fullimg_csv/GSCM_Edge.csv","w")

data_A = list()
data_B = list()
classes = list()

for line in featureA:
    l = line.strip()
    l = l.split(',')
    l = map(float , l)
    data_A.append(l[0:-1])
    classes.append(int(l[-1]))

for line in featureB:
    l = line.strip()
    l = l.split(',')
    l = map(float , l)
    data_B.append(l[0:-1])


for i in range(len(classes)):
    for data in data_A[i]:
        feature.write(str(data)+',')
    for data in data_B[i]:
        feature.write(str(data)+',')
    feature.write(str(classes[i])+'\n')
feature.close()
