import pdb
folder = ""
pathcsv = "data_fullimg_csv/"
patharff = "data_fullimg_arff/"
inpath = pathcsv+folder
outpath = patharff+folder
# names = ["GSCM_1","GSCM_2","GSCM_3","GSCM_4","GSCM_5","GSCM_1_LDA","GSCM_2_LDA","GSCM_3_LDA","GSCM_4_LDA","GSCM_5_LDA","GSCM_all","GSCM_all_LDA"]
names = ["GSCM_Edge","GSCM_Edge_LDA",]
for name in names:
    train_file = open(inpath+name+".csv","r")
    train_data=[]
    train_class=[]
    for line in train_file:
        l = line.strip()
        l = l.split(',')
        l = map(float , l)
        train_data.append(l[0:-1])
        train_class.append(int(l[-1]))

    classnumb = []
    for i in train_class:
        if i not in classnumb:
            classnumb.append(i)
    attribs = len(train_data[0])
    arrfile = open(outpath+name+".arff","w")
    arrfile.write("@RELATION " + name + '\n')
    for i in range(attribs):
        arrfile.write("@ATTRIBUTE a"+str(i)+" NUMERIC\n")
    arrfile.write("@ATTRIBUTE class {")
    for i in range(len(classnumb)-1):
        arrfile.write(str(classnumb[i])+',')

    arrfile.write(str(classnumb[len(classnumb)-1])+"}\n")
    arrfile.write("@DATA\n")

    for i in range(len(train_data)):
        for j in train_data[i]:
            arrfile.write(str(j)+",")
        arrfile.write(str(train_class[i])+"\n")
    arrfile.close()
