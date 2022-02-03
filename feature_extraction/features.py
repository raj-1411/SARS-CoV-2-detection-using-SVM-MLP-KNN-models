from feature_extraction import Gabor
from feature_extraction import GLCM
from feature_extraction import Haralick
import numpy as np


def features_extracton(train_path,features_extraction_method):
    if features_extraction_method == 'gabor':
        return Gabor.gabor(train_path)
    elif features_extraction_method == 'glcm':
        return GLCM.glcm(train_path)
    elif features_extraction_method == 'haralick':
        return Haralick.haralick(train_path)
    else:
        return (np.concatenate((Gabor.gabor(train_path),GLCM.glcm(train_path),Haralick.haralick(train_path)),axis=1))