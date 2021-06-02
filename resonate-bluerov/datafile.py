import numpy as np
import rosbag
import pandas
import matplotlib.pyplot as plt
import tf.transformations
import math

UUV_RADIUS = 0.333
OBSTACLE_RADIUS = 0.5
FAR_ENCOUNTER_RANGE_M = 15.0
CLOSE_ENCOUNTER_RANGE_M = 4.0
COLLISION_RANGE_M = 2.0

MAX_MSG_TIME_SKEW_S = 0.1

UUV_POSE_TOPIC = "/iver0/pose_gt"
# OBSTACLE_POSE_TOPIC = "/spawn_box_obstacles/collision_objects"
OBSTACLE_POSE_TOPIC = "/iver0/box_position"
THRUSTER_DEGRADATION_TOPIC = "/iver0/degradation_gt"


class DataFile(object):
    def __init__(self, filepath):
        # Basic datafile information
        self.filepath = filepath
        self.data = None

        # Various calculated metrics
        # self.am_moving_avg = None
        # self.am_avg_before_top = None
        self.threat_occurred = None
        self.top_occurred = None
        self.consequence_occurred = None
        self.thruster_id = None
        self.thruster_degradation_amount = None
        self.closest_approach = None
        self.closest_approach_index = None

        self._read_bag(self.filepath)
        # self._plot_data()

    def _read_bag(self, filepath):
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

    def plot_data(self):
        fig1 = plt.figure(dpi=300)
        ax1 = fig1.add_subplot(1, 1, 1)
        fig2 = plt.figure(dpi=300)
        ax2 = fig2.add_subplot(1, 1, 1)

        # Plot UUV trajectory, obstacle locations, and point of closest approach
        ax1.plot(self.data["pose_gt"]["x"], self.data["pose_gt"]["y"], label="UUV Trajectory")
        ax1.scatter(self.data["obstacle_pos"]["x"], self.data["obstacle_pos"]["y"], label="Obstacles")
        pca_x = [self.data["pose_gt"]["x"][self.closest_approach_index], self.data["obstacle_pos"]["x"]]
        pca_y = [self.data["pose_gt"]["y"][self.closest_approach_index], self.data["obstacle_pos"]["y"]]
        ax1.plot(pca_x, pca_y, linestyle="--", label="Point of Closest Approach")

        # Fig 1 config
        # Want plot to maintain scale, so set limits the same on X and Y
        lower_lim = min(np.min(self.data["pose_gt"]["y"]), np.min(self.data["pose_gt"]["x"])) - 5
        upper_lim = max(np.max(self.data["pose_gt"]["y"]), np.max(self.data["pose_gt"]["x"])) + 5
        limits = [lower_lim, upper_lim]
        ax1.set_ylim(limits)
        ax1.set_xlim(limits)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend(loc="best")
        ax1.text(limits[0] + 1, limits[1] - 2, "d_min = %.2f" % self.closest_approach)

        # Plot separation distance vs time
        ax2.plot(self.data["pose_gt"]["timestamp"], self.data["separation_dist"], label="Separation Dist")
        ax2.plot(self.data["pose_gt"]["timestamp"], self.data["far_encounter"], label="Far Encounter")
        ax2.plot(self.data["pose_gt"]["timestamp"], self.data["close_encounter"], label="Close Encounter")

        # Fig 2 config
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Distance (m)")
        ax2.legend(loc="best")

        # Show plots
        plt.show()


if __name__ == "__main__":
    # BAGFILE_PATH = "../test/results/recording.bag"
    BAGFILE_PATH = "/home/charlie/alc/bluerov2/resonate/estimation_data/No-Faults/static/run3/task0/recording.bag"
    df = DataFile(BAGFILE_PATH)
    df.plot_data()
