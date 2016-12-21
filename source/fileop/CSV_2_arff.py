import pdb
folder = "Edge/"
pathcsv = "/home/chris/honours/Texture_Analysis/data_block_csv/"
patharff = "/home/chris/honours/Texture_Analysis/data_block_arff/"
inpath = pathcsv+folder
outpath = patharff+folder
# names = ["GSCM_1","GSCM_2","GSCM_3","GSCM_4","GSCM_5","GSCM_all","GSCM_1_LDA","GSCM_2_LDA","GSCM_3_LDA","GSCM_4_LDA","GSCM_5_LDA","GSCM_all_LDA"]
# names = ["Gabor_all", "Gabor_4", "Gabor_8", "Gabor_16", "Gabor_32", "Gabor_all_LDA", "Gabor_4_LDA", "Gabor_8_LDA", "Gabor_16_LDA", "Gabor_32_LDA", ]
names = ["Edge_all","Edge_8","Edge_12","Edge_16","Edge_dp_16","Edge_all_LDA","Edge_8_LDA","Edge_12_LDA","Edge_16_LDA","Edge_dp_16_LDA",]
for name in names:
    train_file = open(inpath+name+".csv","r")
    train_data=[]
    train_class=[]
    for line in train_file:
        l = line.strip()
        l = l.split(',')
        l = list(map(float , l))
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
