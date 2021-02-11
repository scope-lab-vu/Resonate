#!/usr/bin/python3
import textx
import numpy as np
import lxml.etree
import lxml.builder
import sys
import glob
import os
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from textx.metamodel import metamodel_from_file
import utils
import csv


def scenario_town_description(scenarios,id,town):
    scenarios.entities[0].properties[0] = id
    scenarios.entities[0].properties[1] = town

def scenario_weather(scenarios,data_file,num_scenes):
    weather = []
    # if(num_scenes == 0):
    #     scenarios.entities[1].properties[0] = "0.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "0.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "0.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 1):
    #     scenarios.entities[1].properties[0] = "5.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "0.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "0.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 2):
    #     scenarios.entities[1].properties[0] = "5.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "5.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "0.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 3):
    #     scenarios.entities[1].properties[0] = "5.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "5.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "5.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 4):
    #     scenarios.entities[1].properties[0] = "10.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "0.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "0.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 5):
    #     scenarios.entities[1].properties[0] = "10.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "10.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "0.0" #str(np.random.uniform(0,100))
    # elif(num_scenes == 6):
    #     scenarios.entities[1].properties[0] = "10.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] =  "10.0" #str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] =  "10.0" #str(np.random.uniform(0,100))
    # else:
    #     scenarios.entities[1].properties[0] = str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[1] = str(np.random.uniform(0,100))
    #     scenarios.entities[1].properties[2] = str(np.random.uniform(0,100))

    if(num_scenes <= 15):
        scenarios.entities[1].properties[0] = str(np.random.uniform(0,20))
        scenarios.entities[1].properties[1] = str(np.random.uniform(0,20))
        scenarios.entities[1].properties[2] = str(np.random.uniform(0,20))
    elif(num_scenes > 15 and num_scenes < 30):
        scenarios.entities[1].properties[0] = str(np.random.uniform(20,40))
        scenarios.entities[1].properties[1] = str(np.random.uniform(20,40))
        scenarios.entities[1].properties[2] = str(np.random.uniform(20,40))
    elif(num_scenes > 30 and num_scenes < 45):
        scenarios.entities[1].properties[0] = str(np.random.uniform(40,60))
        scenarios.entities[1].properties[1] = str(np.random.uniform(40,60))
        scenarios.entities[1].properties[2] = str(np.random.uniform(40,60))
    elif(num_scenes > 45 and num_scenes < 60):
        scenarios.entities[1].properties[0] = str(np.random.uniform(60,80))
        scenarios.entities[1].properties[1] = str(np.random.uniform(60,80))
        scenarios.entities[1].properties[2] = str(np.random.uniform(60,80))
    else:
        scenarios.entities[1].properties[0] = str(np.random.uniform(80,100))
        scenarios.entities[1].properties[1] = str(np.random.uniform(80,100))
        scenarios.entities[1].properties[2] = str(np.random.uniform(80,100))



    scenarios.entities[1].properties[3] = "0.0"
    scenarios.entities[1].properties[4] = "0.0"
    scenarios.entities[1].properties[5] = "70.0"
    scenarios.entities[1].properties[6] =  "0.0"
    scenarios.entities[1].properties[7] =  "0.0"
    scenarios.entities[1].properties[8] =  "0.0"
    weather.append(str(scenarios.entities[1].properties[0]))
    weather.append(str(scenarios.entities[1].properties[1]))
    weather.append(str(scenarios.entities[1].properties[2]))
    weather.append(str(scenarios.entities[1].properties[3]))
    weather.append(str(scenarios.entities[1].properties[4]))
    weather.append(str(scenarios.entities[1].properties[5]))
    weather.append(str(scenarios.entities[1].properties[6]))
    weather.append(str(scenarios.entities[1].properties[7]))
    weather.append(str(scenarios.entities[1].properties[8]))

    return weather


    # with open(data_file, 'a') as csvfile:
    #     # creating a csv dict writer object
    #     writer = csv.writer(csvfile, delimiter = ' ')
    #     # writing data rows
    #     writer.writerows(weather)

def scenario_agent_route(scenarios,global_route):
    #for i in range(len(global_route)):
            for i in range(len(scenarios.entities[2].properties)):
                for j in range(5):
                    scenarios.entities[2].properties[i] = global_route[i]

            return scenarios

def ego_agent_info(scenarios,global_route):
        scenarios.entities[3].properties[0] = "vehicle.lincoln.mkz2017"
        scenarios.entities[3].properties[1] = "black"
        scenarios.entities[3].properties[2] = "hero"
        scenarios.entities[3].properties[3] = global_route[0]
        return scenarios



if __name__ == '__main__':
        simulation_run = int(input("enter the data file to store the results:"))
        num_scenes = int(input("enter the number of scenes to run  in the simulation:"))
        folder = "/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_routes/simulation%d"%simulation_run #folder to store all the xml generated
        os.makedirs(folder, exist_ok=True)
        data_file = folder+ "/weather_data.csv"
        scenario_meta = metamodel_from_file('entity.tx') #grammer for the scenario language
        scenarios = scenario_meta.model_from_file('scenario.entity') #scenario entities
        global_route,town = utils.parse_routes_file('/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/routes/route_17.xml',False) #global route by reading one route from CARLA AD
        weather_data = []
        for i in range(num_scenes): #Number of different simulation runs to be generated
            scenario_town_description(scenarios,i,town) #town description
            weather = scenario_weather(scenarios,data_file,i) #weather description
            scenario_agent_route(scenarios,global_route) #scenario route to be taken by agent
            ego_agent_info(scenarios,global_route) #ego vehicle info, currently not used
            print("Run%d complete"%i)
            utils.XML_generator(scenarios,i,folder) #generated XML each round
            weather_data.append(weather)
        with open(data_file, 'a') as csvfile:
            # creating a csv dict writer object
            writer = csv.writer(csvfile, delimiter = ',')
            # writing data rows
            writer.writerows(weather_data)
