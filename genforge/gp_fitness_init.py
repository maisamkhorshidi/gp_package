

def gp_fitness_init(gp):
    """Initializes a run."""
    gp.fitness = {
        'values': None,
        'complexities': None,
    }