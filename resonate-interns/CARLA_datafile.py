from datafile import Datafile 
import os
import numpy as np
import fault_modes
import re
import scipy.ndimage
import pandas as pd

DESIRED_BRAKING_MARGIN_FACTOR = 1.3
VEHICLE_LENGTH_M = 4.5
STEP_DELTA_T_S = 0.050
MINIMUM_BRAKING_MARGIN_THRESHOLD = 0
MOVING_AVERAGE_WINDOW_SIZE = 5


class CARLA_Datafile(Datafile):
    def __init__(self):
        super(Datafile, self).__init__()
        self.filename = None
        self.fault_set = None
        self.data = None 
        self.times = None

        # Various calculated metrics
        self.braking_margin = None
        self.braking_threshold_violations = None
        self.collisions = None
        self.am_moving_avg = None
        self.am_avg_before_top = None


    def read(self, filename):
        if self.filename is not None:
            raise RuntimeError("This DataFile object has already loaded a CSV file. Cannot load another file.")

        self.filename = filename
        self.data = pd.read_csv(filename, sep=',')

        
        # FIXME: This is a bit fragile, and highly dependent on a consistent file naming scheme
        # Determine what fault modes were present for each datafile based on the file name
        simple_name, _ = os.path.splitext(os.path.basename(filename))
        # Special case for files named "brightness_XX.csv"
        if simple_name.lower().startswith("brightness"):
            fault_mode = fault_modes.FaultModes.NO_FAULT
        else:

            nums_in_name = [int(x.group()) for x in re.finditer(r'\d+', simple_name)]
    
            # Should be 0 or 1 numbers found in the filename. If 0, assume no fault
            if len(nums_in_name) > 1:
                # raise RuntimeError("Too many numbers in filename: %s" % filename)
                fault_mode = fault_modes.FaultModes.NO_FAULT
            elif len(nums_in_name) == 0:
                fault_mode = fault_modes.FaultModes.NO_FAULT
            else:
                fault_mode = fault_modes.FaultModes(nums_in_name[0])

        
        # Translate fault mode identifier to a set of individual fault modes
        self.fault_set = fault_modes.fault_mode_to_set(fault_mode)

        # Determine times that correspond to data point indices
        self.times = np.arange(len(self.data["Ground Truth"])) * STEP_DELTA_T_S


        # Calculate the "braking margin" and any times where this margin falls below the desired threshold
        self.braking_margin = self.data["Ground Truth"][:] - VEHICLE_LENGTH_M - DESIRED_BRAKING_MARGIN_FACTOR * self.data["stopping_distance"][:]
        self.braking_threshold_violations = (self.braking_margin < MINIMUM_BRAKING_MARGIN_THRESHOLD).astype('int32')

        # Define a collision as a point where the ground truth separation distance is less than the vehicle length
        self.collisions = (self.data["Ground Truth"][:] < VEHICLE_LENGTH_M).astype('int32')

        # For convenience, store flags indicating if a collision/top event has occurred
        self.top_event = self.braking_threshold_violations.max()
        self.consequence = self.collisions.max() > 0

        # Calculate the moving average of the AM Log Martingale value
        self.am_moving_avg = scipy.ndimage.uniform_filter1d(self.data["monitor_result"], MOVING_AVERAGE_WINDOW_SIZE,
                                                            mode="nearest")

        # Find the index of the first Top Event violation and calculate average AM value from [0, idx]
        indices = np.flatnonzero(self.braking_threshold_violations)
        if len(indices) > 0:
            idx = indices[0]
        else:
            idx = len(self.braking_threshold_violations)
        self.am_avg_before_top = np.mean(self.data["monitor_result"][0:idx + 1])


        # add important data to CARLA_Datafile object

        self.independent_vars_cont = {
            "am_average_before_top": self.am_avg_before_top
        }

        self.independent_vars_disc = {
            "NO_FAULT": 0,
            "CENTER_CAM_BLUR": 0,
            "LEFT_CAM_BLUR": 0,
            "RIGHT_CAM_BLUR": 0,
            "LEFT_RIGHT_CAM_BLUR": 0,
            "CENTER_RIGHT_CAM_BLUR": 0,
            "CENTER_LEFT_CAM_BLUR": 0,
            "ALL_CAM_BLUR": 0,
            "CENTER_CAM_OCCLUDE": 0,
            "LEFT_CAM_OCCLUDE": 0,
            "RIGHT_CAM_OCCLUDE": 0,
            "LEFT_RIGHT_CAM_OCCLUDE": 0,
            "CENTER_RIGHT_CAM_OCCLUDE": 0,
            "CENTER_LEFT_CAM_OCCLUDE": 0,
            "ALL_CAM_OCCLUDE": 0,
            "RADAR_FAILURE": 0
        }

        for data in self.fault_set:
            self.independent_vars_disc[data.name] = 1


