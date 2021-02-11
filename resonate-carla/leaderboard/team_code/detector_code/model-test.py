import cv2
import glob
import numpy as np
import scipy as sp
from matplotlib import *
from pylab import *
import time
from keras.models import Model, model_from_json
import numpy as np
from keras import backend as K
import tensorflow as tf
import os
from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import csv


def load_model(model_path):
    with open(model_path + 'auto_model.json', 'r') as jfile:
            model_svdd = model_from_json(jfile.read())
    model_svdd.load_weights(model_path + 'auto_model.h5')
    return model_svdd

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
def load_images(paths,calib_path):
    numImages = 0
    inputs = []
    comp_inp = []
    for path in paths:
        print(path)
        numFiles = len(glob.glob1(path,'*.png'))
        numImages += numFiles
        for img in glob.glob(path+'*.png'):
            img = cv2.imread(img)
            img = cv2.resize(img, (224, 224))
            #img = img / 255.
            inputs.append(img)
    #inpu = shuffle(inputs)
    #print("Total number of images:%d" %(numImages))
    j=0
    for i in range(0,len(inputs),5):
        cv2.imwrite(calib_path + "/frame%d.png"%j,inputs[i])
        j+=1
            #comp_inp.append(calib_path+ "/",inputs[i])
    print("Total number of images:%d" %j)
    # return inputs, comp_inp
    return inputs

def createFolderPaths(path,folders):
    paths = []
    for folder in folders:
        data_path = path + folder + '/'
        paths.append(data_path)
    return paths

def load_training_images(path,trainingFolders,calib_path):
    paths = createFolderPaths(path,trainingFolders)
    return load_images(paths,calib_path)

def load_calib_images(calib_path):
    numImages = 0
    inputs = []
    for img in glob.glob(calib_path+'*.png'):
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        inputs.append(img)
    #inpu = shuffle(inputs)
    print("Total number of images:%d" %(numImages))
    return inputs

def calib_data_generation(model_vae,calib_path,model_path):
    calib_images = load_calib_images(calib_path)
    calib_images = np.array(calib_images)
    calib_images = np.reshape(calib_images, [-1, calib_images.shape[1],calib_images.shape[2],calib_images.shape[3]])
    for i in range(0,len(calib_images)):
        dist_val=[]
        img = np.array(calib_images[i])[np.newaxis]
        predicted_reps = model_vae.predict(img)
        dist = mse(predicted_reps, img)
        print(dist)
        dist_val.append(dist)
        with open(model_path + 'test-noon.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(dist_val)

if __name__ == '__main__':
    path = "/home/scope/Carla/autopilot_Carla_ad/sample_data/new-trial/"
    # list of folders used in training
    trainingFolders = ["clear_noon1","clear_noon2","cloudy_noon1","cloudy_noon2"]
    calib_path = "/home/scope/Carla/autopilot_Carla_ad/sample_data/new-trial/clear_noon1/"
    model_path = "/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/new-trial-100/"
    model_vae=load_model(model_path)
    calib_data_generation(model_vae,calib_path,model_path)
