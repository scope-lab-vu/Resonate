import math
from team_code.risk_calculation.fault_modes import FaultModes
MARTINGALE_THRESHOLD = 6.0
MARTINGALE_SENSITIVITY = 0.2
MARTINGALE_E_FACTOR = math.e ** (MARTINGALE_THRESHOLD * MARTINGALE_SENSITIVITY)

class BowTie(object):
    def __init__(self):
        self.k_b1 = 1.0

    """Simple representation of Bow-Tie Diagram. Placeholder for more general solution"""
    def prob_t1(self, state):
        return 0.5

    def prob_t2(self, state):
        return 0.5

    def prob_b1(self, state):
        martingale = state["monitor_values"]["lec_martingale"]
        p_failure = 1.0 / (1 + MARTINGALE_E_FACTOR/(martingale ** (MARTINGALE_SENSITIVITY / math.log(10, math.e))))
        return 1 - p_failure

    def prob_b2(self, state):
        martingale = state["monitor_values"]["lec_martingale"]
        p_failure = 1.0 / (1 + MARTINGALE_E_FACTOR/(martingale ** (MARTINGALE_SENSITIVITY / math.log(10, math.e))))
        return 1 - p_failure

    def prob_b3(self, state):
        if FaultModes.RADAR_FAILURE in state["fault_modes"]:
            return 0.0

        precip = float(state["enviornment"]["precipitation"]) / 100
        print(precip)
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
        # else:
        #     raise RuntimeError()
