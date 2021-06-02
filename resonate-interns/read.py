import numpy as np
import rosbag
import pandas
import matplotlib.pyplot as plt
import tf.transformations
import math

UUV_POSE_TOPIC = "/iver0/pose_gt"
OBSTACLE_POSE_TOPIC = "/spawn_box_obstacles/collision_objects"
# OBSTACLE_POSE_TOPIC = "/iver0/box_position"
THRUSTER_DEGRADATION_TOPIC = "/iver0/degradation_gt"

UUV_RADIUS = 0.333
OBSTACLE_RADIUS = 0.5
FAR_ENCOUNTER_RANGE_M = 15.0
CLOSE_ENCOUNTER_RANGE_M = 20.0 # can change to 20 temporarily (was 5)
COLLISION_RANGE_M = 10.0 

MAX_MSG_TIME_SKEW_S = 0.1

def read_bag(self, filepath):
    try:
        bag = rosbag.Bag(filepath)
        self.data = {}

        # Read ground-truth position of vehicle
        pose_data = {"x": [], "y": [], "z": [], "orientation": [], "timestamp": []}
        for topic, msg, timestamp in bag.read_messages(UUV_POSE_TOPIC):
            pose_data["x"].append(msg.pose.pose.position.x)
            pose_data["y"].append(msg.pose.pose.position.y)
            pose_data["z"].append(msg.pose.pose.position.z)
            pose_data["orientation"].append(msg.pose.pose.orientation)
            pose_data["timestamp"].append(timestamp.to_sec())
        self.data["pose_gt"] = pandas.DataFrame(data=pose_data)

        # Get positions of any obstacles
        self.data["obstacle_pos"] = []
        obs_pos_data = {"x": [], "y": [], "z": [], "timestamp": []}
        for topic, msg, timestamp in bag.read_messages(OBSTACLE_POSE_TOPIC):
            timestamp = timestamp.to_sec()
            if topic == "/spawn_box_obstacles/collision_objects":
                # Obstacle coords are relative to vehicle. Find vehicle position at this timestamp (closest match)
                pose_timestamps = self.data["pose_gt"]["timestamp"]
                abs_time_diff = np.abs(pose_timestamps - timestamp)
                closest_match_idx = np.argmin(abs_time_diff)
                if abs_time_diff[closest_match_idx] > MAX_MSG_TIME_SKEW_S:
                    raise ValueError("Closest messages exceed maximum allowed time skew.")
                closest_match_pose = self.data["pose_gt"].iloc[closest_match_idx]

                # Store obstacle coords
                quat_msg = closest_match_pose["orientation"]
                x_1 = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 0])
                q_1 = np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w])
                q_1_inv = tf.transformations.quaternion_inverse(q_1)
                x_0 = tf.transformations.quaternion_multiply(tf.transformations.quaternion_multiply(q_1, x_1), q_1_inv)
                obs_pos_data["x"].append(x_0[0] + closest_match_pose["x"])
                obs_pos_data["y"].append(x_0[1] + closest_match_pose["y"])
                obs_pos_data["z"].append(x_0[2] + closest_match_pose["z"])
                obs_pos_data["timestamp"].append(timestamp)

            if topic == "/iver0/box_position":
                # This topic abuses the LatLonDepth message type to store XYZ coordinates in the world frame
                obs_pos_data["x"].append(msg.latitude)
                obs_pos_data["y"].append(msg.longitude)
                obs_pos_data["z"].append(msg.depth)
                obs_pos_data["timestamp"].append(timestamp)
            # FIXME: Assume 1 obstacle for now
            break
        self.data["obstacle_pos"] = pandas.DataFrame(data=obs_pos_data)

        # FIXME: If no static obstacle is found, look for AIS contact

        # Get thruster degradation status
        thruster_efficiency = []
        for topic, msg, timestamp in bag.read_messages(THRUSTER_DEGRADATION_TOPIC):
            self.thruster_id = msg.data[0]
            thruster_efficiency.append(msg.data[1])
        self.data["thruster_efficiency"] = np.array(thruster_efficiency)
        thruster_degraded_indicies = self.data["thruster_efficiency"] < 1.0
        if np.count_nonzero(thruster_degraded_indicies) > 0:
            self.thruster_degradation_amount = 1 - np.average(self.data["thruster_efficiency"][thruster_degraded_indicies])
        else:
            self.thruster_degradation_amount = 0.0

        print("Thruster Degredation:", self.thruster_degradation_amount)

        # Calculate separation distance over the UUV trajectory
        sep_dist = []
        obs_pos = self.data["obstacle_pos"]
        for index, row in self.data["pose_gt"].iterrows():
            # Center-Of-Mass (COM) and Point of Closest Approach (PCA)
            # For PCA, need to consider geometry of UUV and obstacle. Approximated as spheres here.
            com_dist = math.sqrt((row["x"] - obs_pos["x"]) ** 2 +
                                    (row["y"] - obs_pos["y"]) ** 2 +
                                    (row["z"] - obs_pos["z"]) ** 2)
            pca_dist = com_dist - UUV_RADIUS - OBSTACLE_RADIUS
            sep_dist.append(pca_dist)
        self.data["separation_dist"] = np.array(sep_dist)
        self.closest_approach = np.min(self.data["separation_dist"])
        self.closest_approach_index = np.argmin(self.data["separation_dist"])

        # Determine when an encounter has occurred (far and near)
        self.data["far_encounter"] = self.data["separation_dist"] < FAR_ENCOUNTER_RANGE_M
        self.data["close_encounter"] = self.data["separation_dist"] < CLOSE_ENCOUNTER_RANGE_M
        self.data["collision"] = self.data["separation_dist"] < COLLISION_RANGE_M

        # For convenience, store flags indicating if a threat, top event, or consequence has occurred
        self.threat_occurred = np.any(self.data["far_encounter"])
        self.top_occurred = np.any(self.data["close_encounter"])
        self.consequence_occurred = np.any(self.data["collision"])


        # Close bag file after all data is read to save memory
        bag.close()

    except (rosbag.bag.ROSBagFormatException, rosbag.bag.ROSBagException) as e:
        self.data = None
        print("Failed to read ROS Bag file at %s. Omitting." % filepath)