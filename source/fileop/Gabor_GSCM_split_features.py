#----------------------------------------------------------------
# Gabor_path = "/home/chris/honours/Texture_Analysis/data_block_csv/Gabor/"
GSCM_path = "/home/chris/honours/Texture_Analysis/data_block_csv/GSCM/"
train_file = open(GSCM_path + "output.csv","r")
train_data = []
train_class = []
for line in train_file:
	k=line.strip()
	k=k.split(',')
	ky = map(float,k[0:-1])
	train_data.append(ky)
	train_class.append(k[-1])
train_file.close()
#----------------------------------------------------------------
# Gabor_4 = file(Gabor_path + "Gabor_4.csv","w")
# Gabor_8 = file(Gabor_path + "Gabor_8.csv","w")
# Gabor_16 = file(Gabor_path + "Gabor_16.csv","w")
# Gabor_32 = file(Gabor_path + "Gabor_32.csv","w")
# Gabor_all = file(Gabor_path + "Gabor_all.csv","w")

#----------------------------------------------------------------
GSCM_1 = file(GSCM_path + "GSCM_1.csv","w")
GSCM_2 = file(GSCM_path + "GSCM_2.csv","w")
GSCM_3 = file(GSCM_path + "GSCM_3.csv","w")
GSCM_4 = file(GSCM_path + "GSCM_4.csv","w")
GSCM_5 = file(GSCM_path + "GSCM_5.csv","w")
GSCM_all = file(GSCM_path + "GSCM_all.csv","w")
#----------------------------------------------------------------
for i in range(len(train_data)):
    label = str(train_class[i])

    # for data in train_data[i][0:32]:
    #     Gabor_all.write(str(data)+',')
    # Gabor_all.write(label+'\n')
	#
    # #Gabor_4
    # for data in train_data[i][0:8]:
    #     Gabor_4.write(str(data)+',')
    # Gabor_4.write(label+'\n')
	#
    # #Gabor_8
    # for data in train_data[i][8:16]:
    #     Gabor_8.write(str(data)+',')
    # Gabor_8.write(label+'\n')
	#
    # #Gabor_16
    # for data in train_data[i][16:24]:
    #     Gabor_16.write(str(data)+',')
    # Gabor_16.write(label+'\n')
	#
    # #Gabor_32
    # for data in train_data[i][24:32]:
    #     Gabor_32.write(str(data)+',')
    # Gabor_32.write(label+'\n')

#-----------------------------------------------------------------

    for data in train_data[i][0:60]:
        GSCM_all.write(str(data)+',')
    GSCM_all.write(label+'\n')

    #GSCM_1
    for data in train_data[i][0:12]:
        GSCM_1.write(str(data)+',')
    GSCM_1.write(label+'\n')

    #GSCM_2
    for data in train_data[i][12:24]:
        GSCM_2.write(str(data)+',')
    GSCM_2.write(label+'\n')

    #GSCM_3
    for data in train_data[i][24:36]:
        GSCM_3.write(str(data)+',')
    GSCM_3.write(label+'\n')

    #GSCM_4
    for data in train_data[i][36:48]:
        GSCM_4.write(str(data)+',')
    GSCM_4.write(label+'\n')

    #GSCM_5
    for data in train_data[i][48:60]:
        GSCM_5.write(str(data)+',')
    GSCM_5.write(label+'\n')

# Gabor_4.close()
# Gabor_8.close()
# Gabor_16.close()
# Gabor_4.close()
#
GSCM_1.close()
GSCM_2.close()
GSCM_3.close()
GSCM_4.close()
GSCM_5.close()
