from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
from PIL import Image
import numpy as np
import csv
import os
import glob
import pandas as pd
from skimage.transform import resize



def glcm(file_path):
    #GLCM or Gray Level Co-occurence Matrix extracts texture features from images


    PATCH_SIZE = 21

    #GLCM will work on batch of images only if all the images are of same size. 
    #Uncomment the following two lines of code and enter the dimensions of images you want if the dataset has inconsistent sizes of images:

    IMAGE_HEIGHT=2000
    IMAGE_WIDTH=2000

    img_dir = file_path #Enter the directory where all the images are stored
    train_names = os.listdir(img_dir)
    data_path=os.path.join(img_dir,'*g')
    files=sorted(glob.glob(data_path), key = os.path.getmtime)

    eo=len(files)

    img = []
    for f1 in files:
        data = cv2.imread(f1)
        img.append(data)

    print("Extracting GLCM features")
    for i in range(eo):
            img[i] = cv2.cvtColor(img[i] , cv2.COLOR_BGR2GRAY)

            #Uncomment the following line for inconsistent size of images in dataset
            img[i] = cv2.resize(img[i],(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)
            image=img[i]
        
            # select some patches from grassy areas of the image
            grass_locations = [(1000,100), (980,100), (990,120), (985,150)]
            grass_patches = []
            for loc in grass_locations:
                grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,loc[1]:loc[1] + PATCH_SIZE])

            # select some patches from sky areas of the image
            sky_locations = [(417, 415), (427, 413), (420, 410), (422, 412)]
            sky_patches = []
            for loc in sky_locations:
                sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

            # compute some GLCM properties each patch
            xs = []
            ys = []
            bs = []
            cs = []
            ds = []

            for patch in (grass_patches + sky_patches):
                glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
                xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
                ys.append(greycoprops(glcm, 'correlation')[0, 0])
                bs.append(greycoprops(glcm, 'contrast')[0, 0])
                cs.append(greycoprops(glcm, 'energy')[0, 0])
                ds.append(greycoprops(glcm, 'homogeneity')[0, 0])

            temp_xs=xs
            temp_ys=ys
            temp_bs=bs
            temp_cs=cs
            temp_ds=ds

            temp_xs.sort()
            temp_ys.sort()
            temp_bs.sort()
            temp_cs.sort()
            temp_ds.sort()

            xs_max=temp_xs[-1]
            ys_max=temp_ys[-1]
            bs_max=temp_bs[-1]
            cs_max=temp_cs[-1]
            ds_max=temp_ds[-1]

            xs_mean=sum(temp_xs)/len(xs)
            ys_mean=sum(temp_ys)/len(ys)
            bs_mean=sum(temp_bs)/len(bs)
            cs_mean=sum(temp_cs)/len(cs)
            ds_mean=sum(temp_ds)/len(ds)

            xs_var=np.var(np.array(temp_xs))
            ys_var=np.var(np.array(temp_ys))
            bs_var=np.var(np.array(temp_bs))
            cs_var=np.var(np.array(temp_cs))
            ds_var=np.var(np.array(temp_ds))

            df = pd.DataFrame()

            df[0] = [xs_max]
            df[1] = xs_mean
            df[2] = xs_var

            df[3] = ys_max
            df[4] = ys_mean
            df[5] = ys_var

            df[6] = bs_max
            df[7] = bs_mean
            df[8] = bs_var

            df[9] = cs_max
            df[10] = cs_mean
            df[11] = cs_var

            df[12] = ds_max
            df[13] = ds_mean
            df[14] = ds_var
        
            with open('GLCM.csv','a+',newline='') as file:
                writer=csv.writer(file)
            
                writer.writerow([xs_max,xs_mean,xs_var,ys_max,ys_mean,ys_var,bs_max,bs_mean,bs_var,cs_max,cs_mean,cs_var,ds_max,ds_mean,ds_var])
    #Generating numpy array from the csv file created
    print("Extraction succesful!")
    return np.genfromtxt("GLCM.csv",delimiter=',')