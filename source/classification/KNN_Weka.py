# The module performs classification using the Nearest Neghbour classifier on data given
# using 10 fold cross-validation.

################ Imports ################
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, FilteredClassifier, Evaluation
from weka.core.classes import Random
################ Global ################
jvm.start()
data_dir = "/home/chris/honours/Texture_Analysis/data_fullimg_arff/"

names1 = ["GSCM/GSCM_1","GSCM/GSCM_2","GSCM/GSCM_3","GSCM/GSCM_4","GSCM/GSCM_5","GSCM/GSCM_all","GSCM/GSCM_1_LDA","GSCM/GSCM_2_LDA","GSCM/GSCM_3_LDA","GSCM/GSCM_4_LDA","GSCM/GSCM_5_LDA","GSCM/GSCM_all_LDA"]
names2 = ["Gabor/Gabor_all", "Gabor/Gabor_4", "Gabor/Gabor_8", "Gabor/Gabor_16", "Gabor/Gabor_32", "Gabor/Gabor_all_LDA", "Gabor/Gabor_4_LDA", "Gabor/Gabor_8_LDA", "Gabor/Gabor_16_LDA", "Gabor/Gabor_32_LDA", ]
names3 = ["Edge/Edge_all","Edge/Edge_8","Edge/Edge_12","Edge/Edge_16","Edge/Edge_dp_16","Edge/Edge_all_LDA","Edge/Edge_8_LDA","Edge/Edge_12_LDA","Edge/Edge_16_LDA","Edge/Edge_dp_16_LDA",]

results = list()

for filename in ["Edge/Edge_dp_16_LDA"]:
    print("Classifying : " + filename)
    loader = Loader(classname="weka.core.converters.ArffLoader")
    clsf = Classifier(classname="weka.classifiers.lazy.IBk")
    print(clsf.options)
    # clsf.options = ['-K', '1', '-W', '0' , '-A' ,'weka.core.neighboursearch.KDTree -A "weka.core.EuclideanDistance -R first-last" -S weka.core.neighboursearch.kdtrees.SlidingMidPointOfWidestSide -W 0.01 -L 40 -N']
    print(clsf.options)
    fc = FilteredClassifier()
    print("Loading data")
    data = loader.load_file(data_dir + filename + ".arff")
    data.class_is_last()
    # ------------------------------------
    fc.classifier = clsf
    # ------------------------------------
    evl = Evaluation(data)
    print("Evaluating")
    evl.crossvalidate_model(fc, data, 10, Random(1))
    # ------------------------------------
    results.append((filename.split('/')[1], evl.percent_correct))
# ------------------------------------
jvm.stop()
import pdb; pdb.set_trace()
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
