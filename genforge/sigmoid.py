import numpy as np

def sigmoid(input):
    """Compute the sigmoid function for the input."""
    sigmoidval = 1 / (1 + np.exp(-input))
    return sigmoidval
