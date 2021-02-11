#!/usr/bin/env python3
# coding: utf-8

#libraries
import keras
import tensorflow as tf
from keras import backend as K
import cv2
import os
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json, load_model
from keras.layers import Input, Dense
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Lambda, Input, Dense, MaxPooling2D, BatchNormalization,Input
from keras.layers import UpSampling2D, Dropout, Flatten, Reshape, RepeatVector, LeakyReLU,Activation
from keras.callbacks import ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.callbacks import EarlyStopping
keras.callbacks.TerminateOnNaN()
seed = 7
np.random.seed(seed)
from keras.callbacks import CSVLogger
from keras.callbacks import Callback, LearningRateScheduler

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'GPU': 2} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"]="0"#Setting the script to run on GPU:1,2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"#Setting the script to run on GPU:1,2

import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
from sklearn.utils import shuffle

# path to USB
USBPath = "/home/scope/Carla/autopilot_Carla_ad/sample_data/"
# list of folders used in training
trainingFolders = ["clear_noon1","clear_noon2","clear_noon3","clear_noon4","clear_noon5"]
#Only parameters that has to be changed
Working_directory = "/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/trial1/"#working directory
Working_folder = 'left-B-2.0'#experiment
Working_path = Working_directory + Working_folder + '/'
trainfolder = 'train_reconstruction_result'#train folder
data = CSVLogger(Working_path + 'kerasloss.csv', append=True, separator=';')


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

def createFolderPaths(folders):
    paths = []
    for folder in folders:
        path = USBPath + folder + '/' + 'rgb_left_detector' + '/'
        paths.append(path)
    return paths

def load_training_images():
    paths = createFolderPaths(trainingFolders)
    return load_images(paths)


def load_data():
    #Loading images from the datasets
    csv_input = load_training_images()
    len(csv_input)#length of the data
    csv_input = shuffle(csv_input)

    img_train, img_test = np.array(csv_input[0:len(csv_input)-500]), np.array(csv_input[len(csv_input)-500:len(csv_input)])
    img_train = np.reshape(img_train, [-1, img_train.shape[1],img_train.shape[2],img_train.shape[3]])
    img_test = np.reshape(img_test, [-1, img_test.shape[1],img_test.shape[2],img_test.shape[3]])
    #Shuffle the data in order to get different images in train and test datasets.
    #img_train = shuffle(img_train)
    #img_test = shuffle(img_test)
    inp = (img_train, img_test)
    return inp

#Sampling function used by the VAE
def sample_func(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#CNN-VAE model. Only important part is the N_latent variable which holds the latent space data.
def CreateModels(n_latent=40, sample_enc=sample_func, beta=2.0, C=0):
    model = Sequential()
    input_img = Input(shape=(224,224,3), name='image')
    x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1000)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(250)(x)
    x = LeakyReLU(0.1)(x)
#     x = Dense(50)(x)
#     x = LeakyReLU(0.1)(x)

    z_mean = Dense(n_latent, name='z_mean')(x)
    z_log_var = Dense(n_latent, name='z_log_var')(x)
    z = Lambda(sample_func, output_shape=(n_latent,), name='z')([z_mean, z_log_var])

    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()

    latent_inputs = Input(shape=(n_latent,), name='z_sampling')
#     x = Dense(50)(latent_inputs)
#     x = LeakyReLU(0.1)(x)
    x = Dense(250)(latent_inputs)
    x = LeakyReLU(0.1)(x)
    x = Dense(1000)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(3136)(x)
    x = LeakyReLU(0.1)(x)

    x = Reshape((14, 14, 16))(x)

    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(3, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    decoder = Model(latent_inputs, decoded)
    #decoder.summary()

    outputs = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img,outputs)
    #autoencoder.summary()

    def vae_loss(true, pred):
        rec_loss = mse(K.flatten(true), K.flatten(pred))
        rec_loss *= 224*224*3
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(rec_loss + beta*(kl_loss-C))
        return vae_loss
        #autoencoder.add_loss(vae_loss)

    def lr_scheduler(epoch): #learningrate scheduler to adjust learning rate.
        lr = 1e-6
        if epoch > 50:
            print("New learning rate")
            lr = 1e-8
        if epoch > 75:
            print("New learning rate")
            lr = 1e-8
        return lr

    scheduler = LearningRateScheduler(lr_scheduler)
    #Define adam optimizer
    adam = keras.optimizers.Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer='adam',loss=vae_loss, metrics=[vae_loss])

    #Define adam optimizer
    #adam = keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #autoencoder.compile(optimizer='rmsprop',loss=vae_loss, metrics=[vae_loss])
    #autoencoder.compile(optimizer='adam',loss=vae_loss, metrics=[vae_loss])

    return autoencoder, encoder,decoder, z_log_var


#Train function to fit the data to the model
def train(X,autoencoder):
    X_train,X_test = X
    filePath = Working_path + 'weights.best.hdf5'#checkpoint weights

    checkpoint = ModelCheckpoint(filePath, monitor='vae_loss', verbose=1, save_best_only=True, mode='min')
    EarlyStopping(monitor='vae_loss', patience=10, verbose=0),
    callbacks_list = [checkpoint, data]
    autoencoder.fit(X_train, X_train,epochs=75,batch_size=16,shuffle=True,validation_data=(X_test, X_test),callbacks=callbacks_list, verbose=2)

    #checkpoint = ModelCheckpoint(filePath, monitor='vae_loss', verbose=2, save_best_only=True, mode='min')
    #EarlyStopping(monitor='vae_loss', patience=5, verbose=0),
    #es=EarlyStopping(monitor='vae_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    #callbacks_list = [checkpoint, data]
    #autoencoder.fit(X_train, X_train,epochs=2,batch_size=16,shuffle=True,validation_data=(X_test, X_test),callbacks=callbacks_list, verbose=1)


    #Save the autoencoder model
def SaveAutoencoderModel(autoencoder):
	auto_model_json = autoencoder.to_json()
	with open(Working_path + 'auto_model.json', "w") as json_file:
		json_file.write(auto_model_json)
	autoencoder.save_weights(Working_path + 'auto_model.h5')
	print("Saved Autoencoder model to disk")

#Save the encoder model
def SaveEncoderModel(encoder):
	en_model_json = encoder.to_json()
	with open(Working_path + 'en_model.json', "w") as json_file:
		json_file.write(en_model_json)
	encoder.save_weights(Working_path + 'en_model.h5')
	print("Saved Encoder model to disk")

#Test the trained models on a different test data
def test(autoencoder,encoder,test):
    autoencoder_res = autoencoder.predict(test)
    encoder_res = encoder.predict(test)
    res_x = test.copy()
    res_y = autoencoder_res.copy()
    res_x = res_x * 255
    res_y = res_y * 255

    return res_x, res_y, encoder_res

#Save the reconstructed test data in a separate folder.
#For this create a folder named results in the directory you are working in.
def savedata(test_in, test_out, test_encoded, Working_path, trainfolder):
    os.makedirs(Working_path + trainfolder + '/', exist_ok=True)
    for i in range(len(test_in)):
        test_in = np.reshape(test_in,[-1, 224,224,3])#Reshape the data
        test_out = np.reshape(test_out,[-1, 224,224,3])#Reshape the data
        cv2.imwrite(Working_path + trainfolder + '/' + str(i) +'_in.png', test_in[i])
        cv2.imwrite(Working_path + trainfolder + '/' + str(i) +'_out.png', test_out[i])


if __name__ == '__main__':
    print("loading image")
    inp = load_data()
    print("created model")
    autoencoder,encoder,decoder,z_log_var = CreateModels()# Running the autoencoder model
    print("training model")
    train(inp,autoencoder)#Train the model with the data
    print("testing model")
    test_in, test_out, test_encoded = test(autoencoder, encoder, inp[1])#Test the trained model with new data
    print("save model performance")
    savedata(test_in, test_out, test_encoded, Working_path, trainfolder)#Save the data
    print("save encoder model")
    SaveEncoderModel(encoder)
    SaveAutoencoderModel(autoencoder)#Save the autoencoder and encoder models
