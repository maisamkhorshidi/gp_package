import time

def gptic(gp):
    """Updates the running time of this run.
    
    Begins the start timer for the current generation.
    
    Args:
        gp (dict): The genetic programming structure.
    
    Returns:
        dict: The updated genetic programming structure.
    """
    gp['state']['tic'] = time.time()
    return gp
