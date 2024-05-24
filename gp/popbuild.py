import numpy as np
from .utils import selection, mutate, getdepth, getnumnodes, crossover, noise_augmented_boruta

def popbuild(gp):
    """Build next population of individuals."""
    # Initialize new population
    newPop = [[None] * gp.config['runcontrol']['num_pop'] for _ in range(gp.config['runcontrol']['pop_size'])]
    newidx = np.zeros((gp.config['runcontrol']['pop_size'], gp.config['runcontrol']['num_pop']), dtype=int)
    
    # Update gen counter
    gp.state['count'] += 1

    for jj in range(gp.config['runcontrol']['num_pop']):
        for cc in range(1):  # Assuming single class for simplicity, adjust if needed
            # The number of new members to be constructed after elitism is accounted for
            if gp.config['runcontrol']['useboruta']:
                num2boruta = gp.config['runcontrol']['borutafraction']
            else:
                num2boruta = 0

            if gp.config['runcontrol']['num_pop'] > 1:
                num2elite_en = int(np.ceil(gp.config['selection']['elite_fraction_ensemble'][jj] * gp.config['runcontrol']['pop_size']))
            else:
                num2elite_en = 0

            num2elite = int(np.ceil(gp.config['selection']['elite_fraction'][jj] * gp.config['runcontrol']['pop_size']))
            num2build = gp.config['runcontrol']['pop_size'] - num2elite - num2elite_en - num2boruta

            # Parameter shortcuts
            p_mutate = gp.config['operators']['mutation']['p_mutate']
            p_direct = gp.config['operators']['directrepro']['p_direct']
            maxDepth = gp.config['treedef']['max_depth'][jj]
            max_nodes = gp.config['treedef']['max_nodes'][jj]
            p_cross_hi = gp.config['genes']['operators']['p_cross_hi']
            crossRate = gp.config['genes']['operators']['hi_cross_rate']
            useMultiGene = gp.config['genes']['multigene']
            pmd = p_mutate + p_direct
            maxNodesInf = np.isinf(max_nodes)
            maxGenes = gp.config['genes']['max_genes'][jj]

            # Reset cache
            if gp.config['runcontrol']['usecache']:
                gp.fitness['cache'].clear()

            buildCount = 0

            # Loop until the required number of new individuals has been built.
            while buildCount < num2build:
                buildCount += 1
                # Probabilistically select a genetic operator
                p_gen = np.random.rand()
                if p_gen < p_mutate:  # Select mutation
                    eventType = 1
                elif p_gen < pmd:  # Direct reproduction
                    eventType = 2
                else:  # Crossover
                    eventType = 3

                # Mutation
                if eventType == 1:
                    parentIndex = selection(gp, jj)  # Pick the population index of a parent individual using selection operator
                    parent = gp.population[parentIndex][jj][cc]

                    if useMultiGene:  # If using multigene, extract a target gene
                        numParentGenes = len(parent)
                        targetGeneIndex = np.random.randint(0, numParentGenes)
                        targetGene = parent[targetGeneIndex]
                    else:
                        targetGeneIndex = 0
                        targetGene = parent[targetGeneIndex]

                    mutateSuccess = False
                    for _ in range(10):  # Loop until a successful mutation occurs (max loops=10)
                        mutatedGene = mutate(targetGene, gp, jj)
                        mutatedGeneDepth = getdepth(mutatedGene)
                        if mutatedGeneDepth <= maxDepth:
                            if maxNodesInf:
                                mutateSuccess = True
                                break
                            mutatedGeneNodes = getnumnodes(mutatedGene)
                            if mutatedGeneNodes <= max_nodes:
                                mutateSuccess = True
                                break

                    # If no success then use parent gene
                    if not mutateSuccess:
                        mutatedGene = targetGene

                    # Add the mutated individual to new pop
                    parent[targetGeneIndex] = mutatedGene
                    newPop[buildCount][jj][cc] = parent
                    newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndex][jj]

                # Direct reproduction
                elif eventType == 2:
                    parentIndex = selection(gp, jj)  # Pick a parent
                    parent = gp.population[parentIndex][jj][cc]

                    # Copy to new population
                    newPop[buildCount][jj][cc] = parent
                    newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndex][jj]

                    # Store fitness etc of copied individual if cache enabled
                    if gp.config['runcontrol']['usecache']:
                        cachedData = {
                            'complexity': gp.fitness['complexity'][parentIndex],
                            'returnvalues': gp.fitness['returnvalues'][parentIndex],
                            'value': gp.fitness['values'][parentIndex]
                        }
                        gp.fitness['cache'][buildCount] = cachedData

                # Crossover operator
                elif eventType == 3:
                    highLevelCross = False
                    if useMultiGene:
                        # Select crossover type if multigene enabled
                        if np.random.rand() < p_cross_hi:
                            highLevelCross = True

                    # Select parents
                    parentIndex = selection(gp, jj)
                    dad = gp.population[parentIndex][jj][cc]
                    numDadGenes = len(dad)
                    parentIndexd = parentIndex

                    parentIndex = selection(gp, jj)
                    mum = gp.population[parentIndex][jj][cc]
                    numMumGenes = len(mum)
                    parentIndexm = parentIndex

                    if highLevelCross and (numMumGenes > 1 or numDadGenes > 1):
                        # High level crossover
                        dadGeneSelectionInds = np.random.rand(numDadGenes) < crossRate
                        mumGeneSelectionInds = np.random.rand(numMumGenes) < crossRate

                        if not np.any(dadGeneSelectionInds):
                            dadGeneSelectionInds[np.random.randint(numDadGenes)] = True

                        if not np.any(mumGeneSelectionInds):
                            mumGeneSelectionInds[np.random.randint(numMumGenes)] = True

                        dadSelectedGenes = [gene for idx, gene in enumerate(dad) if dadGeneSelectionInds[idx]]
                        mumSelectedGenes = [gene for idx, gene in enumerate(mum) if mumGeneSelectionInds[idx]]

                        dadRemainingGenes = [gene for idx, gene in enumerate(dad) if not dadGeneSelectionInds[idx]]
                        mumRemainingGenes = [gene for idx, gene in enumerate(mum) if not mumGeneSelectionInds[idx]]

                        mumOffspring = mumRemainingGenes + dadSelectedGenes
                        dadOffspring = dadRemainingGenes + mumSelectedGenes

                        # Curtail offspring longer than the max allowed number of genes
                        newPop[buildCount][jj][cc] = mumOffspring[:maxGenes]
                        newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndexm][jj]
                        buildCount += 1

                        if buildCount < num2build:
                            newPop[buildCount][jj][cc] = dadOffspring[:maxGenes]
                            newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndexd][jj]
                    else:
                        # Low level crossover
                        if useMultiGene:
                            dad_target_gene_num = np.random.randint(numDadGenes)
                            mum_target_gene_num = np.random.randint(numMumGenes)
                            dad_target_gene = dad[dad_target_gene_num]
                            mum_target_gene = mum[mum_target_gene_num]
                        else:
                            dad_target_gene_num = 0
                            mum_target_gene_num = 0
                            dad_target_gene = dad[dad_target_gene_num]
                            mum_target_gene = mum[mum_target_gene_num]

                        for _ in range(10):  # Loop (max 10 times) until both children meet size constraints
                            # Produce 2 offspring
                            son, daughter = crossover(mum_target_gene, dad_target_gene, gp)
                            son_depth = getdepth(son)

                            # Check if both children meet size and depth constraints
                            if son_depth <= maxDepth:
                                daughter_depth = getdepth(daughter)
                                if daughter_depth <= maxDepth:
                                    if maxNodesInf:
                                        crossOverSuccess = True
                                        break
                                    son_nodes = getnumnodes(son)
                                    if son_nodes <= max_nodes:
                                        daughter_nodes = getnumnodes(daughter)
                                        if daughter_nodes <= max_nodes:
                                            crossOverSuccess = True
                                            break

                        # If no success then re-insert parents
                        if not crossOverSuccess:
                            son = dad_target_gene
                            daughter = mum_target_gene

                        # Write offspring back to right gene positions in parents and write to population
                        dad[dad_target_gene_num] = son
                        newPop[buildCount][jj][cc] = dad
                        newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndexd][jj]
                        buildCount += 1

                        if buildCount < num2build:
                            mum[mum_target_gene_num] = daughter
                            newPop[buildCount][jj][cc] = mum
                            newidx[buildCount][jj] = gp.class_['idx'][gp.state['count']-1][parentIndexm][jj]

            # Skim off the existing elite individuals and stick them on the end of the new population
            # Get indices of best num2skim individuals
            sortIndex = np.argsort(gp.class_['fitnessindiv'][:, jj])
            if not gp.fitness['minimisation']:
                sortIndex = sortIndex[::-1]

            for g in range(num2elite):
                oldIndex = sortIndex[g]
                if g == 0:
                    bestInds = np.where(gp.class_['fitnessindiv'][:, jj] == gp.class_['fitnessindiv'][oldIndex, jj])[0]
                    oldIndex = bestInds[np.argsort(gp.class_['complexity'][bestInds, jj])[0]]

                newIndex = num2build + g
                copiedIndividual = gp.population[oldIndex][jj][cc]

                if gp.config['runcontrol']['usecache']:
                    cachedData = {
                        'complexity': gp.fitness['complexity'][oldIndex],
                        'returnvalues': gp.fitness['returnvalues'][oldIndex],
                        'value': gp.fitness['values'][oldIndex]
                    }
                    gp.fitness['cache'][newIndex] = cachedData

                newPop[newIndex][jj][cc] = copiedIndividual
                newidx[newIndex][jj] = gp.class_['idx'][gp.state['count']-1][oldIndex][jj]

            # Stick the individuals with best ensemble fitness at the end of the new population
            if num2elite_en > 0:
                sortIndex = np.argsort(gp.class_['fitness_validation_ensemble'])
                if not gp.fitness['minimisation']:
                    sortIndex = sortIndex[::-1]

                for g in range(num2elite_en):
                    oldIndex = sortIndex[g]
                    if g == 0:
                        bestInds = np.where(gp.class_['fitness_validation_ensemble'] == gp.class_['fitness_validation_ensemble'][oldIndex])[0]
                        oldIndex = bestInds[np.argsort(gp.class_['complexity'][bestInds, jj])[0]]

                    newIndex = num2build + num2elite + g
                    copiedIndividual = gp.population[oldIndex][jj][cc]

                    if gp.config['runcontrol']['usecache']:
                        cachedData = {
                            'complexity': gp.fitness['complexity'][oldIndex],
                            'returnvalues': gp.fitness['returnvalues'][oldIndex],
                            'value': gp.fitness['values'][oldIndex]
                        }
                        gp.fitness['cache'][newIndex] = cachedData

                    newPop[newIndex][jj][cc] = copiedIndividual
                    newidx[newIndex][jj] = gp.class_['idx'][gp.state['count']-1][oldIndex][jj]

            if num2boruta > 0:
                sortIndex = np.argsort(gp.class_['fitnessindiv'][:, jj])
                if not gp.fitness['minimisation']:
                    sortIndex = sortIndex[::-1]

                newIndex = num2build + num2elite + num2elite_en
                oldIndex = sortIndex[0]
                tourId = [sortIndex[0]] + list(np.random.choice(sortIndex[1:], num2boruta-1, replace=False))
                popout = noise_augmented_boruta(tourId, jj, gp)

                newPop[newIndex][jj][cc] = popout
                newidx[newIndex][jj] = gp.class_['idx'][gp.state['count']-1][oldIndex][jj]

    gp.population = newPop
    gp.class_['idx'][gp.state['count']] = newidx
    return gp
