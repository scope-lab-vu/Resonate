import matplotlib.pyplot as plt
import numpy as np
import sigmoid

SIGMOID_PARAMETERS = [0.18428081, 4.34419664]
SIGMOID_PARAMETERS_REALLOCATION = [5.73672305e+02, 7.07026933e-04]

# Calculate sigmoid values over desired x-range
x = np.linspace(0, 0.6, 100)
y1 = sigmoid.bounded_sigmoid(x, *SIGMOID_PARAMETERS)
y2 = sigmoid.bounded_sigmoid(x, *SIGMOID_PARAMETERS_REALLOCATION)

plt.rcParams['font.size'] = 12
fig1 = plt.figure(dpi=300)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(x, y1, label="No Reallocation")
ax1.plot(x, y2, linestyle="--", label="Reallocation")
ax1.set_ylim([0, 1])
ax1.legend(loc="upper left")
ax1.set_xlabel("Thruster Degradation Level")
ax1.set_ylabel("Probability of Barrier Failure")
ax1.set_title("Barrier Effectiveness vs. Thruster Degradation")
plt.show()
