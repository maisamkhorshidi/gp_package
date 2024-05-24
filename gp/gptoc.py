import time

def gptoc(gp):
    """Updates the running time of this run.
    
    Args:
        gp (dict): The genetic programming structure.
    
    Returns:
        dict: The updated genetic programming structure.
    """
    elapsed_time = time.time() - gp['state']['tic']
    gp['state']['runTimeElapsed'] += elapsed_time
    return gp
