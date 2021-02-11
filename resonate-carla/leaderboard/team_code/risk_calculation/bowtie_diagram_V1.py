import math
from team_code.risk_calculation.fault_modes import FaultModes, fault_mode_to_set,SingularFaultModes
from team_code.risk_calculation import sigmoid

MARTINGALE_THRESHOLD = 6.0
MARTINGALE_SENSITIVITY = 0.5
MARTINGALE_E_FACTOR = math.e ** (MARTINGALE_THRESHOLD * MARTINGALE_SENSITIVITY)
SIGMOID_PARAMETERS = [35.4970268,  0.37144841]

class BowTie(object):
    def __init__(self):
        self.k_b1 = 1.0

    """Simple representation of Bow-Tie Diagram. Placeholder for more general solution"""
    def rate_t1(self, state):
        return 1.0

    def rate_t2(self, state):
        return 4.0

    def prob_b1(self, state,fault_modes):
        prob = 1.0
        fault_set = fault_mode_to_set(fault_modes)
        if SingularFaultModes.NO_FAULT in fault_set or len(fault_set) == 0:
            prob *= 0.833

        if SingularFaultModes.CENTER_CAM_BLUR in fault_set and state["monitor_values"]["center_blur_dect"] == True:
            prob *= 0.351

        if SingularFaultModes.LEFT_CAM_BLUR in fault_set and state["monitor_values"]["left_blur_dect"] == True:
            prob *= 0.378

        if SingularFaultModes.RIGHT_CAM_BLUR in fault_set and state["monitor_values"]["right_blur_dect"] == True:
            prob *= 0.270

        if SingularFaultModes.CENTER_CAM_OCCLUDE in fault_set and state["monitor_values"]["center_occ_dect"] == True:
            prob *= 0.342

        if SingularFaultModes.LEFT_CAM_OCCLUDE in fault_set and state["monitor_values"]["left_occ_dect"] == True:
            prob *= 0.395

        if SingularFaultModes.RIGHT_CAM_OCCLUDE in fault_set and state["monitor_values"]["right_occ_dect"] == True:
            prob *= 0.395

        log_martingale = state["monitor_values"]["lec_martingale"]
        p_failure = sigmoid.bounded_sigmoid(log_martingale, *SIGMOID_PARAMETERS)
        prob *= (1 - p_failure)

        return prob

        # else:
        #     prob=1.0
        #     return prob
            #log_martingale = state["monitor_values"]["lec_martingale"]
            #p_failure = 1.0 / (1 + MARTINGALE_E_FACTOR * (math.e ** (-MARTINGALE_SENSITIVITY * log_martingale)))
            #return 1 - p_failure


    def prob_b2(self, state,fault_modes):
        prob = 1.0
        fault_set = fault_mode_to_set(fault_modes)
        if SingularFaultModes.NO_FAULT in fault_set or len(fault_set) == 0:
            prob *= 0.833

        if SingularFaultModes.CENTER_CAM_BLUR in fault_set and state["monitor_values"]["center_blur_dect"] == True:
            prob *= 0.351

        if SingularFaultModes.LEFT_CAM_BLUR in fault_set and state["monitor_values"]["left_blur_dect"] == True:
            prob *= 0.378

        if SingularFaultModes.RIGHT_CAM_BLUR in fault_set and state["monitor_values"]["right_blur_dect"] == True:
            prob *= 0.270

        if SingularFaultModes.CENTER_CAM_OCCLUDE in fault_set and state["monitor_values"]["center_occ_dect"] == True:
            prob *= 0.342

        if SingularFaultModes.LEFT_CAM_OCCLUDE in fault_set and state["monitor_values"]["left_occ_dect"] == True:
            prob *= 0.395

        if SingularFaultModes.RIGHT_CAM_OCCLUDE in fault_set and state["monitor_values"]["right_occ_dect"] == True:
            prob *= 0.395
            
        log_martingale = state["monitor_values"]["lec_martingale"]
        p_failure = sigmoid.bounded_sigmoid(log_martingale, *SIGMOID_PARAMETERS)
        prob *= (1 - p_failure)

        return prob

    def prob_b3(self, state):
        if FaultModes.RADAR_FAILURE in state["fault_modes"]:
            return 0.0

        precip = float(state["enviornment"]["precipitation"]) / 100
        if 0.0 < precip <= 0.20:
            return 0.857143
        elif precip <= 0.40:
            return 0.750000
        elif precip <= 0.60:
            return 0.730769
        elif precip <= 0.80:
            return 0.0
        elif precip <= 1.0:
            return 0.0
        else:
            raise RuntimeError()
