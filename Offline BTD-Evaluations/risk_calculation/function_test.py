from bowtie_diagram import BowTie
import matplotlib.pyplot as plt


EXAMPLE_MONITOR_VALUES = [x for x in range(-5, 21)]

bowtie = BowTie()
state = {"monitor_values": {"lec_martingale": None}}
true_y_vals = []
true_x_vals = []
for x_val in EXAMPLE_MONITOR_VALUES:
    true_x_vals.append(x_val)
    state["monitor_values"]["lec_martingale"] = x_val
    true_y_vals.append(bowtie.prob_b1(state))


plt.scatter(true_x_vals, true_y_vals)
plt.xlabel("Log Martingale")
plt.ylabel("P(B1 | S)")
plt.show()