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



def extract_run_path(path):
    runs_path = []
    for run_data in glob.glob(path +"/run*.csv"):
        runs_path.append(run_data)
    #runs_path.sort(reverse=True)
    # for i in range(1,runs+1):
    #     runs_path.append(path + 'run%d.csv'%i)
    #print(runs_path)
    return runs_path

def extract_collision_data(path):
    collision_path = []
    for collision_data in glob.glob(path +"/data*.txt"):
        collision_path.append(collision_data)
    collision_path.sort(reverse=False)
    print(collision_path)
    final_colisions = []
    for j in range(len(collision_path)):
        file1 = open(collision_path[j], 'r')
        Lines = file1.readlines()
        count = 0
        data = []
        collisions = []
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
        for x in range(len(collisions)):
            col.append("".join(collisions[x]))
        final_colisions.append(col)
    #print(final_colisions)

    return final_colisions


def extract_fault_data(fault_data_path):
    fault_data = []
    with open(fault_data_path, 'r') as readFile:
        reader = csv.reader(readFile)
        next(reader)
        for row in reader:
            data = []
            data.append(row[0])
            if(int(row[0])==1):
                data1 = row[1].strip().split(',')
                data1[0] = data1[0][1:]
                data1[len(data1)-1]=data1[len(data1)-1][:-1]
                data2 = row[2].strip().split(',')
                data2[0] = data2[0][1:]
                data2[len(data2)-1]=data2[len(data2)-1][:-1]
                data3 = row[3].strip().split(',')
                data3[0] = data3[0][1:]
                data3[len(data3)-1]=data3[len(data3)-1][:-1]
                data.append(float(data1[0])/20)
                data.append(float(data1[0])/20 + float(data2[0])/20)
                data.append(data3[0])
                fault_data.append(data)
            if(int(row[0])> 1):
                data1 = row[1].strip().split(',')
                data1[0] = data1[0][1:]
                data1[len(data1)-1]=data1[len(data1)-1][:-1]
                data2 = row[2].strip().split(',')
                data2[0] = data2[0][1:]
                data2[len(data2)-1]=data2[len(data2)-1][:-1]
                data3 = row[3].strip().split(',')
                data3[0] = data3[0][1:]
                data3[len(data3)-1]=data3[len(data3)-1][:-1]
                data.append(float(data1[0])/20)
                data.append(float(data1[len(data1)-1])/20)
                data.append(float(data1[0])/20 + float(data2[0])/20)
                data.append(float(data1[len(data2)-1])/20 + float(data2[len(data2)-1])/20)
                data.append(data3[0])
                data.append(data3[len(data3)-1])
                fault_data.append(data)
        #print(fault_data)
    return fault_data

def extract_weather_data(weather_path):
    weather_data = []
    with open(weather_path, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            weather_data.append(row)
    #print(weather_data)
    return weather_data

def plot(runs_path,weather_data,collision_times,fault_data,l,path):
        risk = []
        mval = []
        steps = []
        time = []
        with open(runs_path, 'r') as readFile:
            reader = csv.reader(readFile)
            next(reader)
            for row in reader:
                steps.append(float(row[0]))
                time.append(float(row[0])/20.0)
                risk.append(float(row[2]))
                mval.append(float(row[1]))

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Risk Score', color=color)
        ax1.set_ylim([0, 1])
        x=0
        cycol = cycle('bgrcmk')
        if(len(collision_times)!=0):
            for xc in collision_times:
                ax1.axvline(x=float(xc),linewidth = 2, linestyle ="--", color ='green', label="collision" if x == 0 else "")
                x+=1
        if(fault_data[0]=="1"):
            ax1.axvspan(fault_data[1],fault_data[2], alpha=0.2, color = 'yellow', label = "fault %s"%fault_data[3])
        if(fault_data[0] > "1"):
            ax1.axvspan(fault_data[1],fault_data[3], alpha=0.2, color = next(cycol), label = "fault %s"%fault_data[5])
            ax1.axvspan(fault_data[2],fault_data[4], alpha=0.2, color = next(cycol), label = "fault %s"%fault_data[6])
        ax1.plot(time, risk, color=color, label= 'risk')
        ax1.tick_params(axis='y', labelcolor=color)

        #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        #color = 'tab:blue'
        #ax2.set_ylabel('Monitor_result', color=color)
        #ax2.plot(time, mval, color=color, label = 'monitor results')
        #ax2.set_ylim([-5, 40])
        #ax2.tick_params(axis='y', labelcolor=color)
        #ax2.set_title("Scene with Cloud:%s, Precip:%s, Precip-deposit:%s"%(weather_data[4],weather_data[2],weather_data[3]))
        fig.legend(loc=8, bbox_to_anchor=(0.5, -0.02),fancybox=True, shadow=True, ncol=4)
        #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)
        fig.subplots_adjust(bottom=0.5)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(path+'/run%d.png'%(l), bbox_inches='tight')
        plt.cla()
    #plt.show()


#if __name__ == '__main__':
    # #collision_times = [[20,46],[37],[17,20,42,45,46]]
    # runs = int(input("Enter the simulation run to be plotted:"))
    # path = "/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/simulation%d/"%runs
    # weather_path =  path + "simulation_data.csv" #"/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/simulation%d/simulation_data.csv"%runs
    # fault_data_path = path + "fault_data.csv" #"/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/simulation%d/fault_data.csv"%runs
    # collision_times = extract_collision_data(path)
    # fault_data = extract_fault_data(fault_data_path)
    # runs_path = extract_run_path(path)
    # weather_data = extract_weather_data(weather_path)
    # for i in range(len(runs_path)):
    #          plot(runs_path[i],weather_data[i+1],collision_times[i],fault_data[i])
