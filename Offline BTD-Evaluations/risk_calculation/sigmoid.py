import numpy as np

SIGMOID_LOWER_BOUND = 10**-15
SIGMOID_UPPER_BOUND = 1 - SIGMOID_LOWER_BOUND


# Sigmoid function
def bounded_sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    y = np.maximum(y, SIGMOID_LOWER_BOUND)
    y = np.minimum(y, SIGMOID_UPPER_BOUND)
    return y
