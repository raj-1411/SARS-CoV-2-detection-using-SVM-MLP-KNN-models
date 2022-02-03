import cv2
import numpy as np
import os
import glob
import mahotas as mt
import csv


def haralick(file_path):
    
    # function to extract haralick textures from an image
    def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean  = textures.mean(axis=0)
        return ht_mean

    # load the training dataset
    train_path  = file_path #Enter the directory where all the images are stored
    train_names = os.listdir(train_path)


    # empty list to hold feature vectors and train labels
    train_features = []
    train_labels   = []

    # loop over the training dataset
    print ("Extracting Haralick textures..")
    cur_path = os.path.join(train_path, '*g')
    cur_label = train_names
    i = 0
    with open('Haralick.csv','a+',newline='') as obj:
                    writer = csv.writer(obj)
                    for file in sorted(glob.glob(cur_path), key=os.path.getmtime):
                        #read the training image
                        image=cv2.imread(file)

                        #convert the image to grayscale
                        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        
                        #extract haralick texture from image
                        features=extract_features(gray)
                        #print(features)
                        
                        #append the feature vector and label
                        train_features.append(features)
                        train_labels.append(cur_label[i])

                        
                        writer.writerow(features)

                        #show loop update
                        i+=1

        
    # have a look at the size of our feature vector and labels
    #Generating numpy array from the csv file created
    print("Extraction succesful!")
    return np.genfromtxt("Haralick.csv",delimiter=',')