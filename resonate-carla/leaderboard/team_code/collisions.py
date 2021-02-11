import cv2
import glob
import numpy as np
import scipy as sp
from matplotlib import *
from pylab import *
import time
import os
from sklearn.utils import shuffle
from keras.models import model_from_json
from keras.losses import mse
import csv
import pandas as pd
import re
from itertools import cycle
import scipy.integrate as integrate
import pandas as pd
import math
from natsort import natsorted


def extract_collision_data(path):
    collision_path = []
    for collision_data in glob.glob(path +"data*.txt"):
        collision_path.append(collision_data)
    collision_path = natsorted(collision_path)
    print(collision_path)
    final_colisions = []
    for j in range(len(collision_path)):
        file1 = open(collision_path[j], 'r')
        Lines = file1.readlines()
        count = 0
        data = []
        collisions = []
        Frames = []
        col = []
        for line in Lines:
                data.append(line.strip())
        for i in range(len(data)):
            number = []
            if(data[i]!= "" and data[i][0].isdigit()):
                for k in range(len(data[i])):
                    if(data[i][k].isdigit()):
                        number.append(data[i][k])
                    elif(data[i][k] == " "):
                        break
                collisions.append(number)

        for i in range(len(data)):
            number = []
            if(data[i]!= "" and data[i][0] =='D' ):
                for k in range(len(data[i])):
                    if(data[i][k].isdigit()):
                        number.append(data[i][k])
                    elif(data[i][k] == " "):
                        break
                Frames.append(number)


        for x in range(len(collisions)):
            col.append("".join(collisions[x]))
        final_colisions.append(col)
        no_repeat_collisions = []
        final = []
        final_colisions_no_repeat = []


    for i in range(len(final_colisions)):
        final_colisions_no_repeat.append([])
        final.append([])

    for i in range(len(final_colisions)):
        for j in range(len(final_colisions[i])):
            if final_colisions[i][j] not in final_colisions_no_repeat[i]:
                final_colisions_no_repeat[i].append(final_colisions[i][j])

    print(final_colisions)
    print(final_colisions_no_repeat)
    #print(final)


    return final_colisions, final_colisions_no_repeat


if __name__ == '__main__':
    runs = int(input("Enter the simulation run to be plotted:"))
    path = "/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/new-run1/simulation%d/"%runs
    collision_times, no_repeat_collisions = extract_collision_data(path)
