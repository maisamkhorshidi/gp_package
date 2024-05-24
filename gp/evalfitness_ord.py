import numpy as np
from .utils import tree2evalstr, getcomplexity, getnumnodes

def evalfitness_ord(gp):
    """Evaluate the fitness of individuals (ordinary version)."""
    clsNum = 1  # Assuming single class for simplicity, adjust if needed
    popNum = gp.config['runcontrol']['num_pop']
    popSize = gp.config['runcontrol']['pop_size']
    evalstrs = [[None] * popNum for _ in range(popSize)]
    for i in range(popSize):
        for jj in range(popNum):
            evalstrs[i][jj] = [None] * clsNum
    
    complexityMeasure = gp.config['fitness']['complexityMeasure']
    complexities = np.zeros(popSize)
    complexitiesindiv = np.zeros((popSize, popNum))
    fitfun = gp.config['fitness']['fitfun']
    fitvals = np.full(popSize, np.inf)
    returnvals = [None] * popSize

    usecache = gp.config['runcontrol']['usecache']

    for i in range(popSize):
        # Update state to index of the individual that is about to be evaluated
        gp.state['current_individual'] = i

        if usecache and i in gp.fitness['cache']:
            cache = gp.fitness['cache'][i]
            complexities[i] = cache['complexity']
            fitvals[i] = cache['value']
            returnvals[i] = cache['returnvalues']
        else:
            for jj in range(popNum):
                for kk in range(clsNum):
                    # Process coded trees into evaluable expressions
                    evalstrs[i][jj][kk] = tree2evalstr(gp.population[i][jj][kk], gp)
                    # Store complexity of individual
                    if complexityMeasure:
                        complexitiesindiv[i, jj] += getcomplexity(gp.population[i][jj][kk])
                        complexities[i] += getcomplexity(gp.population[i][jj][kk])
                    else:
                        complexitiesindiv[i, jj] += getnumnodes(gp.population[i][jj][kk])
                        complexities[i] += getnumnodes(gp.population[i][jj][kk])

            evalstr1 = evalstrs[i]
            fitness, gp = fitfun(evalstr1, gp)
            returnvals[i] = gp.fitness['returnvalues'][i]
            fitvals[i] = fitness

    # Attach returned values to original GP structure
    gp.fitness['values'] = fitvals
    gp.fitness['returnvalues'] = returnvals
    gp.fitness['complexity'] = complexities
    gp.class_['complexity'] = complexitiesindiv
    gp.state['current_individual'] = popSize

    return gp
