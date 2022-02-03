import cv2
import os
import glob
import numpy as np 
import matplotlib.pyplot as plt
import csv



def gabor(file_path):

    img_dir = file_path # Enter Directory where all the images are stored
    train_names = os.listdir(img_dir)
    data_path = os.path.join(img_dir,'*g')
    files = sorted(glob.glob(data_path), key = os.path.getmtime)

    eo=len(files)

    img = []
    for f1 in files:
        data = cv2.imread(f1)
        img.append(data)

    gamma=0.5
    sigma=0.56
    theta_list=[0, np.pi, np.pi/2, np.pi/4, 3*np.pi/4] #Angles
    phi=0
    lamda_list=[2*np.pi/1, 2*np.pi/2, 2*np.pi/3, 2*np.pi/4, 2*np.pi/5] #wavelengths
    num=1

    print("Extracting Gabor features")
    with open('Gabor.csv','a+',newline='') as file:
        writer=csv.writer(file)
    
        for i in range(eo):
            img[i] = cv2.cvtColor(img[i] , cv2.COLOR_BGR2GRAY)
            local_energy_list=[]
            mean_ampl_list=[]
        
            for theta in theta_list:
                for lamda in lamda_list:
                    kernel=cv2.getGaborKernel((3,3),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)
                    fimage = cv2.filter2D(img[i], cv2.CV_8UC3, kernel)
                
                    mean_ampl=np.sum(abs(fimage))
                    mean_ampl_list.append(mean_ampl)
                
                    local_energy=np.sum(fimage**2)
                    local_energy_list.append(local_energy)
                
                    num+=1
            writer.writerow(local_energy_list+mean_ampl_list)

    #Generating numpy array from the csv file created
    print("Extraction succesful!")
    return np.genfromtxt("Gabor.csv",delimiter=',')