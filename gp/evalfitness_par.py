import numpy as np
from multiprocessing import Pool
from .utils import tree2evalstr, getcomplexity, getnumnodes

def evalfitness_par(gp):
    """Evaluate the fitness of individuals (parallel version)."""
    popNum = gp.config['runcontrol']['num_pop']
    popSize = gp.config['runcontrol']['pop_size']
    evalstrs = [[None] * popNum for _ in range(popSize)]
    complexityMeasure = gp.config['fitness']['complexityMeasure']
    complexities = np.zeros(popSize)
    fitfun = gp.config['fitness']['fitfun']
    fitvals = np.zeros(popSize)
    returnvals = [None] * popSize

    usecache = gp.config['runcontrol']['usecache']

    def evaluate_individual(i):
        tempgp = gp.copy()  # Deep copy to simulate independent workers

        # Update state of temp variable to index of the individual that is about to be evaluated
        tempgp.state['current_individual'] = i

        if usecache and i in tempgp.fitness['cache']:
            cache = tempgp.fitness['cache'][i]
            complexities[i] = cache['complexity']
            fitvals[i] = cache['value']
            returnvals[i] = cache['returnvalues']
        else:
            for jj in range(popNum):
                # Process coded trees into evaluable expressions
                evalstrs[i][jj] = tree2evalstr(tempgp.population[i][jj], gp)

                # Store complexity of individual
                if complexityMeasure:
                    complexities[i] += getcomplexity(tempgp.population[i][jj])
                else:
                    complexities[i] += getnumnodes(tempgp.population[i][jj])

                fitness, tempgp = fitfun(evalstrs[i], tempgp)
                returnvals[i] = tempgp.fitness['returnvalues'][i]
                fitvals[i] = fitness

    with Pool() as pool:
        pool.map(evaluate_individual, range(popSize))

    # Attach returned values to original GP structure
    gp.fitness['values'] = fitvals
    gp.fitness['returnvalues'] = returnvals
    gp.fitness['complexity'] = complexities
    gp.state['current_individual'] = popSize

    return gp
