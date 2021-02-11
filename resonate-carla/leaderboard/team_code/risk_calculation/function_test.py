from bowtie_diagram import BowTie
import matplotlib.pyplot as plt
import math


EXAMPLE_MONITOR_VALUES = [20.01, 45.23, 120.96, 12973.2, 412398.82, 1235820982.23, 9812356.276, 76312.2, 2123.1, 1812356.1]

bowtie = BowTie()
state = {"monitor_values": {"lec_martingale": None}}
y_vals = []
x_vals = []
for x_val in EXAMPLE_MONITOR_VALUES:
    x_vals.append(math.log(x_val, 10))
    state["monitor_values"]["lec_martingale"] = x_val
    y_vals.append(bowtie.prob_b1(state))

plt.scatter(x_vals, y_vals)
plt.show()