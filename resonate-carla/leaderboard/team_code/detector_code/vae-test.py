#!/usr/bin/env python
# coding: utf-8
import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
import psutil
from sklearn.utils import shuffle
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.activations import linear
from keras.models import Model, model_from_json
import numpy as np
from keras.callbacks import Callback, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error

def load_model(model_path):
    with open(model_path + 'auto_model.json', 'r') as jfile:
            model_svdd = model_from_json(jfile.read())
    model_svdd.load_weights(model_path + 'auto_model.h5')
    return model_svdd

#Load complete input images without shuffling
def load_images(paths):
    numImages = 0
    inputs = []
    for path in paths:
        numFiles = len(glob.glob1(path,'*.png'))
        numImages += numFiles
        for img in glob.glob(path+'*.png'):
            img = cv2.imread(img)
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            inputs.append(img)
    #inpu = shuffle(inputs)
    print("Total number of images:%d" %(numImages))
    return inputs

def createFolderPaths(train_data_path, train_folders):
    paths = []
    for folder in train_folders:
        path = train_data_path + folder + '/'
        paths.append(path)
    return paths

def load_training_images(train_data_path, train_folders):
    paths = createFolderPaths(train_data_path, train_folders)
    return load_images(paths)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

#Load complete input images without shuffling
def load_training_images1(train_data):
    inputs = []
    comp_inp = []
    with open(train_data + 'calibration.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img = cv2.imread(train_data + row[0])
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                inputs.append(img)
            return inputs

def vae_prediction(model_vae,test_data_path,test_folders):
    print("==============PREDICTING THE LABELS ==============================")
    #X_validate =  load_training_images(test_data_path, test_folders)
    X_validate = load_training_images(test_data_path,test_folders)
    X_validate = np.array(X_validate)
    X_validate = np.reshape(X_validate, [-1, X_validate.shape[1],X_validate.shape[2],X_validate.shape[3]])
    anomaly=0
    tval = []
    dist_val=[]
    for i in range(0,len(X_validate)):
        val=[]
        anomaly_val = 0
        t1 = time.time()
        img = np.array(X_validate[i])[np.newaxis]
        predicted_reps = model_vae.predict(img)
        dist = mse(predicted_reps, img)
        cpu = psutil.cpu_percent()
        t2 = time.time()-t1
        print(dist)
        dist_val.append(dist)
        # gives an object with many fields
        #mem = psutil.virtual_memory().total / (1024.0 ** 3)#virtual memory stats
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss/ (1000.0 ** 3)
        #dist = np.sum(((predicted_reps.any() - img.any()) ** 2), axis=1)
        if(dist > 7.5): #where 10.0 is the threshold.
            anomaly_val = 1
            anomaly+=1

        tval.append(t2)
        val.append(anomaly_val)
        #val.append(cpu)
        #val.append(frame_time)

        with open('/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/SVDD/tcps-evaluation/vae-illumination-change-light.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(val)
    print(anomaly)

    print(sum(tval)/len(X_validate))
    print(max(dist_val))


if __name__ == '__main__':
    test_data_path =  "/home/scope/Carla/CARLA_0.9.6/PythonAPI/new/dataset/"     #SVDD/data-generator/" #"/home/scope/Carla/CARLA_0.9.6/PythonAPI/CarlaData/"
    test_folders = ["new-road"]
    model_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/VAE/train-illumination/" #path to save the svdd weights
    model_vae=load_model(model_path)
    vae_prediction(model_vae,test_data_path,test_folders)
