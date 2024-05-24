import numpy as np
from .treegen import treegen
from .utils import getnumnodes

def initbuild(gp):
    """Generate an initial population of GP individuals."""
    clsNum = 1  # Assuming 1 class for simplicity, adjust if needed
    popNum = gp.config['runcontrol']['num_pop']
    popSize = gp.config['runcontrol']['pop_size']
    maxNodes = gp.config['treedef']['max_nodes']
    maxGenes = gp.config['genes']['max_genes']

    # Override any gene settings if using single gene gp
    if not gp.config['genes'].get('multigene', True):
        maxGenes = [1] * popNum

    # Initialize vars
    gp.population = [[None] * popNum for _ in range(popSize)]
    numGenes = 1

    # Building process
    for jj in range(popNum):
        for i in range(popSize):
            gp.population[i][jj] = [None] * clsNum
            for kk in range(clsNum):
                # Loop through population and randomly pick number of genes in individual
                if maxGenes[jj] > 1:
                    numGenes = np.random.randint(1, maxGenes[jj] + 1)

                individ = [None] * numGenes  # Construct empty individual

                for z in range(numGenes):  # Loop through genes in each individual and generate a tree for each
                    geneLoopCounter = 0
                    while True:
                        geneLoopCounter += 1
                        # Generate a trial tree for gene z
                        temp = treegen(gp, jj)
                        numnodes = getnumnodes(temp)

                        if numnodes <= maxNodes[jj]:
                            copyDetected = False
                            if z > 1:  # Check previous genes for copies
                                for j in range(z):
                                    if temp == individ[j]:
                                        copyDetected = True
                                        break

                            if not copyDetected:
                                break

                        # Display a warning if having difficulty building trees due to constraints
                        if not gp.config['runcontrol'].get('quiet', True) and geneLoopCounter > 10:
                            print('initbuild: iterating tree build loop because of uniqueness constraints.')

                    individ[z] = temp

                # Write new individual to population cell array
                gp.population[i][jj][kk] = individ

    return gp
