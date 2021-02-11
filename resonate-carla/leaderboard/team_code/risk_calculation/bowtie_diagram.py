import math
from team_code.risk_calculation.fault_modes import FaultModes, fault_mode_to_set,SingularFaultModes
from team_code.risk_calculation import sigmoid


class BowTie(object):
    def __init__(self):
        """Simple representation of Bow-Tie Diagram. Placeholder for more general solution"""
        # self.b1_sigmoid_parameters = [35.4970268, 0.37144841]
        self.b1_sigmoid_parameters = [5.75415274, 0.04937048]
        # Table mapping failure modes to probabilities P(B1 | FM) and P(B1 | !FM)
        self.b1_cond_prob_table = {"center_blur_dect":  (0.351, 0.417),
                                   "left_blur_dect":    (0.378, 0.409),
                                   "right_blur_dect":   (0.270, 0.443),
                                   "center_occ_dect":   (0.342, 0.421),
                                   "left_occ_dect":     (0.395, 0.404),
                                   "right_occ_dect":    (0.395, 0.404)}


    def rate_t1(self, state):
        return 1.0

    def rate_t2(self, state):
        return 4.0

    def prob_b1(self, state,fault_modes):
        base_prob = 0.4
        prob = base_prob
        fault_set = fault_mode_to_set(fault_modes)
        for detector_name, (p_true, p_false) in self.b1_cond_prob_table.items():
            if state["monitor_values"][detector_name]:
                prob *= p_true
            else:
                prob *= p_false

            # if state["monitor_values"]["center_blur_dect"] == True and SingularFaultModes.CENTER_CAM_BLUR in fault_set:
            #     prob *= p_true
            # else:
            #     prob *= p_false
            # if SingularFaultModes.LEFT_CAM_BLUR in fault_set and state["monitor_values"]["left_blur_dect"] == True:
            #     prob *= p_true
            # else:
            #     prob *= p_false
            # if SingularFaultModes.RIGHT_CAM_BLUR in fault_set and state["monitor_values"]["right_blur_dect"] == True:
            #     prob *= p_true
            # else:
            #     prob *= p_false
            # if SingularFaultModes.CENTER_CAM_OCCLUDE in fault_set and state["monitor_values"]["center_occ_dect"] == True:
            #     prob *= p_true
            # else:
            #     prob *= p_false
            # if SingularFaultModes.LEFT_CAM_OCCLUDE in fault_set and state["monitor_values"]["left_occ_dect"] == True:
            #     prob *= p_true
            # else:
            #     prob *= p_false
            # if SingularFaultModes.RIGHT_CAM_OCCLUDE in fault_set and state["monitor_values"]["right_occ_dect"] == True:
            #     prob *= p_true
            # else:
            #     prob *= p_false

            prob /= base_prob
        # Similarly for AM value, multiply total probability by P(B1 | AM) / P(B1)
        log_martingale = state["monitor_values"]["lec_martingale"]
        p_failure = sigmoid.bounded_sigmoid(log_martingale, *self.b1_sigmoid_parameters)
        prob *= (1 - p_failure)
        prob /= base_prob

        return prob


    def prob_b2(self, state,fault_modes):
        return self.prob_b1(state,fault_modes)

    def prob_b3(self, state):
        if FaultModes.RADAR_FAILURE in state["fault_modes"]:
            return 0.0

        precip = float(state["enviornment"]["precipitation"]) / 100
        if 0.0 < precip <= 0.20:
            return 0.833333
        elif precip <= 0.40:
            return 0.700000
        elif precip <= 0.60:
            return 0.714286
        elif precip <= 0.80:
            return 0.041667
        elif precip <= 1.0:
            return 0.055556
        else:
            raise RuntimeError()
