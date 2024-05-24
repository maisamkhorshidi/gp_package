import numpy as np

def argmax(input_array):
    """Get the indices of the maximum values along an axis."""
    return np.argmax(input_array, axis=1)
