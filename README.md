# SARS-CoV-2-detection-using-SVM-MLP-KNN-models

## Project Description
This is a python-based project classifying the covid and non-covid lung CT-Scan images by extracting the traditional features, namely, Gabor, GLCM and Haralick features from the SARS-COV-2 CT-Scan dataset and applying the SVM, KNN and MLP prediction models on the extracted features, giving a different accuracy and confusion matrix in each case.

## Dataset description
SARS-CoV-2 CT scan dataset is a publicly available dataset, containing 1252 CT scans that are positive for SARS-CoV-2 infection (COVID-19) and 1230 CT scans for patients non-infected by SARS-CoV-2, 2482 CT scans in total. These data have been collected from real patients in hospitals from Sao Paulo, Brazil. The aim of this dataset is to encourage the research and development of artificial intelligent methods which are able to identify if a person is infected by SARS-CoV-2 through the analysis of his/her CT scans. The dataset is available at:
www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset

## Features extracted
In this project, three traditional feature extraction methods have been used to extract the features from the dataset, namely:  
- `Gabor`  
-	`Gray-Level Co-occurrence Matrix (GLCM)`  
-	`Haralick`  
Gabor is a linear filter used for texture analysis, which analyses whether there is any specific frequency content in a specific direction.  
GLCM represents the second order statistical information of gray levels between neighboring pixels in an image.  
Haralick texture features are calculated from Gray-Level Co-Occurrence Matrix (GLCM). They are common texture descriptors in image analysis.

## Classifier models used
Three classifier models have been applied on the features extracted, namely:  
-	`Support Vector Machines (SVM) (RBF Kernel)`  
-	`K-Nearest Neighbors (KNN) (K=2 used)`  
-	`Multi-Layer Perceptron (MLP)`  
A generalized application of these three models yield results which gives an accuracy in the descending order MLP, KNN, SVM.

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `pandas`, `OpenCV`, and `scikit-learn`. All these dependencies can be installed just by the following command line argument
- `pip install requirements.txt`

## Code implementation
- ### Data paths :
      Current directory -----> data
                                 |
                                 |
                                 |               
                                 ----------> train and test 
                                                   |
                                            ------- -------
                                            |             |
                                            V             V
                                          images        labels
                               
- Where the folder `images` contains original images in `.jpg`/`.png` format and the folder `labels` contains corresponding labels in `.csv` format.   
- `Note:`The `.csv` file containing labels must have 'labels' keyword as the heading of the repective column.                                          
                                          
                                       
- ### Training model with Traditional Features :
      -help

      optional arguments:
        -h, --help            show this help message and exit
        -tr TR_PATH, --tr_path TR_PATH
                              Path to the train data
        -la LA_PATH, --la_path LA_PATH
                              Path to the label data
        -featr FEATR_TYPE, --featr_type FEATR_TYPE
                              Type of feature selection
        -model_type MODEL_TYPE, --model_type MODEL_TYPE
                              Type of training model selection
        
-  ### Run the following for training and validation :
  
      `python main.py -tr data/train/images/ -la data/train/labels/label.csv -featr gabor  -model_type svm`
