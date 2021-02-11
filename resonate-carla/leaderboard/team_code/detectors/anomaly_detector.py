import cv2
import numpy as np
from keras.models import Model, model_from_json
import numpy as np
from keras import backend as K
import tensorflow as tf
import os
from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import csv

def occlusion_detector(image,threshold):
    """Determines occlusion percentage and returns
       True for occluded or False for not occluded"""

    # Create mask and find black pixels on image
    # Color all found pixels to white on the mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[np.where((image <= [15,15,15]).all(axis=2))] = [255,255,255]

    # Count number of white pixels on mask and calculate percentage
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    percentage = (cv2.countNonZero(mask)/ (w * h)) * 100
    #print("occlusion%f"%percentage)
    if percentage > threshold:
        return percentage, True
    else:
        return percentage, False

#image = cv2.imread('2.jpg')
#percentage, occluded = detect_occluded(image)
#print('Pixel Percentage: {:.2f}%'.format(percentage))
#print('Occluded:', occluded)

def blur_detector(image, threshold=20):
    """
    Determines if an image is blur
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    #print("Blur%f"%fm)
    if fm < threshold:
        return fm, True
    else:
        return fm, False

def integrand(k,p_anomaly):
    result = 1.0
    for i in range(len(p_anomaly)):
        result *= k*(p_anomaly[i]**(k-1.0))
    return result

def assurance_monitor(image,model,calibration_set,pval_queue,sval_queue):
    p_anomaly = pval_queue.get()
    prev = sval_queue.get()
    prev_value = []
    anomaly=0
    m=0
    sliding_window = 20
    threshold = 10.0
    img = np.array(image)[np.newaxis]
    predicted_reps = model_vae.predict(img)
    dist = mse(predicted_reps, img)
    for i in range(len(calibration_set)):
        if(dist>calibration_set[i]):
            anomaly+=1
    p_value = anomaly/len(calibration_set)
    #pval.append(p_value)
    #final_pval = sum(pval)/len(pval)
    #anomaly_val.append(final_pval)
    #time_val.append(float(i*0.1))
    if(p_value<0.005):
        p_anomaly.append(0.005)
    else:
        p_anomaly.append(p_value)
    if(len(p_anomaly))>= sliding_window:
        p_anomaly = p_anomaly[-1*sliding_window:]
    m = integrate.quad(integrand,0.0,1.0,args=(p_anomaly))
    m_val = round(math.log(m[0]),2)
    print(m_val)
    M.append(math.log(m[0]))
    p_value = round(p_value,2)
    if(i==0):
        S = 0
        S_prev = 0
    else:
        S = max(0, prev[0]+prev[1]-delta)
    S_prev = S
    m_prev = m[0]
    prev_value.append(S_prev)
    prev_value.append(m_prev)
    #state_val.append(S)
    if(S > threshold):
        val=1
    else:
        val=0
    pval_queue.put(p_anomaly)
    sval_queue.put(prev_value)
