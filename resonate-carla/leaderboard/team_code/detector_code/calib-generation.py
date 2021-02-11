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
from keras.losses import mse
#from sklearn.metrics import mean_squared_error
import csv
#from skimage.measure import structural_similarity as ssim
from skimage import measure

os.environ["CUDA_VISIBLE_DEVICES"]="2"#Setting the script to run on GPU:1,2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# #Load complete input images without shuffling
# def load_training_images(train_data,save_path):
#     inputs = []
#     comp_inp = []
#     num_images = len(sorted(glob.glob(train_data+ "/*.png")))
#     print(num_images)
#     images = []
#     for img in sorted(glob.glob(train_data+ "/*.png")):
#         img = cv2.imread(img)
#         img = cv2.resize(img, (256, 144))
#         print(img.shape)
#         images.append(img)
#     i=0
#     for I in images:
#         i+=1
#         cv2.imwrite(save_path+"/frame%d.png"%i,I)
#
# # path to USB
# USBPath = "/home/scope/Carla/training_data/"
# # list of folders used in training
# trainingFolders = ["rgb1","rgb2"]
# #Only parameters that has to be changed
# Working_directory = "/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/"#working directory
# Working_folder = 'trial2'#experiment
# Working_path = Working_directory + Working_folder + '/'
# trainfolder = 'train_reconstruction_result'#train folder
# data = CSVLogger(Working_path + 'kerasloss.csv', append=True, separator=';')

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
    for i in range(0,len(inputs),4):
        cv2.imwrite(calib_path + "/frame%d.png"%j,inputs[i])
        j+=1
            #comp_inp.append(calib_path+ "/",inputs[i])
    print("Total number of images:%d" %j)
    # return inputs, comp_inp
    return inputs

def createFolderPaths(path,folders):
    paths = []
    for folder in folders:
        data_path = path + folder + '/' + 'rgb_detector' + '/'
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
        #dist = mse(predicted_reps, img)
        dist = np.square(np.subtract(np.array(predicted_reps),img)).mean()
        print(dist)
        dist_val.append(dist)
        with open(model_path + 'calibration_center.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(dist_val)

if __name__ == '__main__':
    path = "/home/scope/Carla/autopilot_Carla_ad/sample_data/"
    # list of folders used in training
    trainingFolders = ["run3","run4"]#["clear_noon1","clear_noon2","clear_noon3","clear_noon4","clear_noon5"]
    calib_path = "/home/scope/Carla/autopilot_Carla_ad/sample_data/calibration_new_center/"
    model_path = "/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/trial1/new-B-1.2/"
    load_training_images(path,trainingFolders,calib_path)
    model_vae=load_model(model_path)
    calib_data_generation(model_vae,calib_path,model_path)
