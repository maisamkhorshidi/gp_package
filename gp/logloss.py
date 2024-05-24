import numpy as np

def logloss(prob, y):
    """Compute the log loss."""
    num_data = prob.shape[0]
    loss = np.sum(np.sum(-1. / num_data * np.log(prob) * y))
    return loss
