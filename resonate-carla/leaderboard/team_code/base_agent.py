#Entry point to the AV. Includes the sensors to be attached to the AV
#Camera faults (Blur and occlusion) and adverse scene (brightness) is introduced in this script
#Blur - OpenCV blur, Occlusion- OpenCV based continuous black pixels, Brightness- OpenCV based brightness in range (35,50)
import time
import csv
import cv2
import carla
import os
from leaderboard.autoagents import autonomous_agent
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights
from team_code.planner import RoutePlanner
from srunner.scenariomanager.carla_data_provider import CarlaActorPool
import numpy as np
import random
import utils

#Fault list hypothesis. This acts as our diagnosis
def get_fault_list(fault_type):
    if(fault_type == 4):
        return (2,3)
    if(fault_type == 5):
        return (1,3)
    if(fault_type == 6):
        return (1,2)
    if(fault_type == 7):
        return (1,2,3)
    if(fault_type == 11):
        return (9,10)
    if(fault_type == 12):
        return (8,10)
    if(fault_type == 13):
        return (8,9)
    if(fault_type == 14):
        return (8,9,10)
    else:
        return fault_type

class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file,data_folder,route_folder,k,model_path):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.data_folder = data_folder
        self.scene_number = k
        self.filename =    self.data_folder + "/fault_data.csv" #file to save fault information
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.fault_type_list = None
        self.brightness_present = random.randint(0,1) # introduce brightness no --> 0, yes --> 1
        self.fault_scenario =   random.randint(0,2) # number of faults none --> 0
        self.fault_step = []
        self.fault_time = []
        self.fault_type = []
        self.fault_type_list = []
        for x in range(self.fault_scenario): #loop to select random fault durations and time of fault occurance
            self.num = x*600
            self.fault_step.append(random.randint(100+self.num,150+self.num))
            self.fault_time.append(random.randint(100,125))

        print(self.fault_step)
        print(self.fault_time)
        self.fields = ['fault_scenario',
                        'fault_step',
                        'fault_time',
                        'fault_type',
                        'fault_list',
                        'brightness_value'
                        ]

        if(self.fault_scenario == 0):
            if(self.brightness_present == 1):
                self.brightness_value = random.randint(35,50) #randomly generated brightness intensity
                print("Brightness:%d"%self.brightness_value)
            else:
                self.brightness_value = 0
            self.fault_type.append(0)
            self.fault_step.append(0)
            self.fault_time.append(0)
            self.fault_type_list = -1
            #self.brightness_value = 0
        elif(self.fault_scenario >= 1):
            if(self.brightness_present == 1):
                self.brightness_value = random.randint(35,50)
                print("Brightness:%d"%self.brightness_value)
            else:
                self.brightness_value = 0
            for x in range(self.fault_scenario):
                self.fault_type.append(12)    #(random.randint(8,14))
                self.fault_type_list.append(get_fault_list(self.fault_type))
        print(self.fault_type)

        self.dict = [{'fault_scenario':self.fault_scenario,'fault_step':self.fault_step,'fault_time':self.fault_time,'fault_type':self.fault_type,'fault_list':self.fault_type_list,'brightness_value':self.brightness_value}]

        file_exists = os.path.isfile(self.filename)

        # writing to csv file
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = self.fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.dict)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self._vehicle = CarlaActorPool.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self._traffic_lights = list()

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 2.8, 'y': 0.0, 'z': 1.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'fov': 25,
                    'sensor_tick': 0.05,
                    'id': 'radar'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 2.8, 'y': 0.25, 'z': 1.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'fov': 25,
                    'sensor_tick': 0.05,
                    'id': 'radar_right'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 'y': 0.0, 'z': 100.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'map'
                    }

                ]

    #Function to add randomly added faults to the camera images
    def add_fault(self,rgb,rgb_left,rgb_right,points,gps,fault_type):
        if(fault_type == 0):
            print("No Fault")
        if(fault_type == 1):
            rgb = cv2.blur(rgb,(10,10))
        elif(fault_type == 2):
            rgb_left = cv2.blur(rgb_left,(10,10))
        elif(fault_type == 3):
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(fault_type == 4):
            rgb_left = cv2.blur(rgb_left,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(fault_type == 5):
            rgb = cv2.blur(rgb,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(fault_type == 6):
            rgb_left = cv2.blur(rgb_left,(10,10))
            rgb = cv2.blur(rgb,(10,10))
        elif(fault_type == 7):
            rgb_left = cv2.blur(rgb_left,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
            rgb = cv2.blur(rgb,(10,10))
        if(fault_type == 8):
            h, w, _ = rgb.shape
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 9):
            h, w, _ = rgb_left.shape
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 10):
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 11):
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 12):
            #print("Right & center Camera Images Occluded")
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 13):
            h, w, _ = rgb.shape
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 14):
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(fault_type == 15):
            noise = np.random.normal(0, .1, points.shape)
            points += noise
        elif(fault_type == 16):
            noise = np.random.normal(0, .001, gps.shape)
            gps += noise

        return rgb, rgb_left, rgb_right, points, gps

    #Sensor data entry
    def tick(self, input_data):
        self.step += 1
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        radar_data = (input_data['radar'][1])
        points = np.reshape(radar_data, (len(radar_data), 4))
        radar_data_right = (input_data['radar_right'][1])
        points_right = np.reshape(radar_data_right, (len(radar_data_right), 4))

        #Add brightness at 250th step when adverse scene chosen
        if(self.step > 250):
            hsv = cv2.cvtColor(rgb_right, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            lim = 255 - self.brightness_value
            v[v > lim] = 255
            v[v <= lim] += self.brightness_value
            final_hsv = cv2.merge((h, s, v))
            rgb_right = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            lim = 255 - self.brightness_value
            v[v > lim] = 255
            v[v <= lim] += self.brightness_value
            final_hsv = cv2.merge((h, s, v))
            rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            hsv = cv2.cvtColor(rgb_left, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            lim = 255 - self.brightness_value
            v[v > lim] = 255
            v[v <= lim] += self.brightness_value
            final_hsv = cv2.merge((h, s, v))
            rgb_left = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        #Synthetically add faults based on the chosen number of faults
        if(self.fault_scenario == 1):
            if (self.step > self.fault_step[0] and self.step<self.fault_step[0] + self.fault_time[0]):
                    rgb,rgb_left,rgb_right,points,gps=self.add_fault(rgb,rgb_left,rgb_right,points,gps,self.fault_type[0])
        elif(self.fault_scenario > 1):
            if (self.step > self.fault_step[0] and self.step<self.fault_step[0] + self.fault_time[0]):
                    rgb,rgb_left,rgb_right,points,gps=self.add_fault(rgb,rgb_left,rgb_right,points,gps,self.fault_type[0])
            if (self.step > self.fault_step[1] and self.step<self.fault_step[1] + self.fault_time[1]):
                    rgb,rgb_left,rgb_right,points,gps=self.add_fault(rgb,rgb_left,rgb_right,points,gps,self.fault_type[1])


        return {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'cloud':points,
                'cloud_right':points_right,
                'fault_scenario': self.fault_scenario,
                'fault_step': self.fault_step,
                'fault_duration': self.fault_time,
                'fault_type': self.fault_type,
                }
