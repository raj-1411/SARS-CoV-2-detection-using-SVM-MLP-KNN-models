from feature_extraction import features
from classifier_list import classifier
import argparse
import os



parser = argparse.ArgumentParser(description = 'Training model with Traditional Features')
# Paths
parser.add_argument('-tr','--tr_path',type=str, 
                    default = 'data/train/images/', 
                    help = 'Path to the train data')
parser.add_argument('-la','--la_path',type=str, 
                    default = 'data/train/labels/', 
                    help = 'Path to the label data')                    
parser.add_argument('-featr','--featr_type',type=str, 
                    default = '#', 
                    help = 'Type of feature selection')
parser.add_argument('-model_type','--model_type',type=str, 
                    default = '#', 
                    help = 'Type of training model selection')



args = parser.parse_args()
train_path = args.tr_path
label_path = args.la_path
featr = args.featr_type
model = args.model_type



features_raw = features.features_extracton(train_path,featr)
classifier.classifier_model(features_raw,label_path,model)
if os.path.exists("Gabor.csv"):
      os.remove("Gabor.csv")
if os.path.exists("GLCM.csv"):
      os.remove("GLCM.csv")
if os.path.exists("Haralick.csv"):
      os.remove("Haralick.csv")