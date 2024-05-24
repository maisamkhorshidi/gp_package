import numpy as np
from .utils import getcomplexity, getnumnodes, tree2evalstr

def updatestats(gp):
    nanInd = np.isnan(gp['fitness']['values'])
    
    if gp['runcontrol']['useboruta']:
        borInd = gp['runcontrol']['pop_size'] - 1
        if gp['fitness']['minimisation']:
            if gp['state']['count'] == 1:
                gp['results']['history']['borutafitness'][gp['state']['count'] - 1] = gp['class']['fitness_validation_ensemble'][borInd]
            else:
                gp['results']['history']['borutafitness'][gp['state']['count'] - 1] = min(gp['class']['fitness_validation_ensemble'][borInd], gp['results']['history']['borutafitness'][gp['state']['count'] - 2])
        else:
            if gp['state']['count'] == 1:
                gp['results']['history']['borutafitness'][gp['state']['count'] - 1] = gp['class']['fitness_validation_ensemble'][borInd]
            else:
                gp['results']['history']['borutafitness'][gp['state']['count'] - 1] = max(gp['class']['fitness_validation_ensemble'][borInd], gp['results']['history']['borutafitness'][gp['state']['count'] - 2])
        
        borindiv = gp['pop'][borInd]
        gp['results']['history']['borutacomplexity'] = 0
        gp['results']['history']['borutanodecount'] = 0
        for i in range(len(borindiv)):
            for j in range(len(borindiv[i])):
                gp['results']['history']['borutacomplexity'] += getcomplexity(borindiv[i][j])
                gp['results']['history']['borutanodecount'] += getnumnodes(borindiv[i][j])
    
    if gp['fitness']['minimisation']:
        gp['fitness']['values'][nanInd] = np.inf
        bestTrainFitness = np.min(gp['fitness']['values'])
    else:
        infInd = np.isinf(gp['fitness']['values'])
        gp['fitness']['values'][infInd] = -np.inf
        gp['fitness']['values'][nanInd] = -np.inf
        bestTrainFitness = np.max(gp['fitness']['values'])
    
    bestTrainInds = np.where(gp['fitness']['values'] == bestTrainFitness)[0]
    nodes = gp['fitness']['complexity'][bestTrainInds]
    ind = np.argmin(nodes)
    bestTrainInd = bestTrainInds[ind]

    gp['state']['best'] = {
        'fitness': bestTrainFitness,
        'individual': gp['pop'][bestTrainInd],
        'returnvalues': gp['fitness']['returnvalues'][bestTrainInd],
        'complexity': 0,
        'nodecount': 0
    }

    for i in range(len(gp['state']['best']['individual'])):
        for j in range(len(gp['state']['best']['individual'][i])):
            gp['state']['best']['complexity'] += getcomplexity(gp['state']['best']['individual'][i][j])
            gp['state']['best']['nodecount'] += getnumnodes(gp['state']['best']['individual'][i][j])

    if gp['fitness']['minimisation']:
        if gp['state']['count'] == 1:
            gp['results']['history']['bestfitness'][gp['state']['count'] - 1] = bestTrainFitness
        else:
            gp['results']['history']['bestfitness'][gp['state']['count'] - 1] = min(bestTrainFitness, gp['results']['history']['bestfitness'][gp['state']['count'] - 2])
    else:
        if gp['state']['count'] == 1:
            gp['results']['history']['bestfitness'][gp['state']['count'] - 1] = bestTrainFitness
        else:
            gp['results']['history']['bestfitness'][gp['state']['count'] - 1] = max(bestTrainFitness, gp['results']['history']['bestfitness'][gp['state']['count'] - 2])
    
    gp['results']['history']['bestcomplexity'][gp['state']['count'] - 1] = gp['state']['best']['complexity']
    gp['state']['best']['index'] = bestTrainInd

    notinfInd = ~np.isinf(gp['fitness']['values'])
    gp['state']['meanfitness'] = np.mean(gp['fitness']['values'][notinfInd])
    gp['state']['std_devfitness'] = np.std(gp['fitness']['values'][notinfInd])
    gp['results']['history']['meanfitness'][gp['state']['count'] - 1] = gp['state']['meanfitness']
    gp['results']['history']['std_devfitness'][gp['state']['count'] - 1] = gp['state']['std_devfitness']
    gp['results']['history']['meancomplexity'][gp['state']['count'] - 1] = np.mean(gp['fitness']['complexity'][notinfInd])
    gp['results']['history']['std_devcomplexity'][gp['state']['count'] - 1] = np.std(gp['fitness']['complexity'][notinfInd])

    if gp['state']['count'] == 1:
        gp['results']['best'] = {
            'fitness': gp['state']['best']['fitness'],
            'individual': gp['state']['best']['individual'],
            'returnvalues': gp['state']['best']['returnvalues'],
            'complexity': gp['state']['best']['complexity'],
            'nodecount': gp['state']['best']['nodecount'],
            'foundatgen': 0,
            'eval_individual': [[tree2evalstr(ind, gp) for ind in ind_list] for ind_list in gp['state']['best']['individual']]
        }
    else:
        updateTrainBest = False

        if gp['fitness']['complexityMeasure']:
            stateComp = gp['state']['best']['complexity']
            bestComp = gp['results']['best']['complexity']
        else:
            stateComp = gp['state']['best']['nodecount']
            bestComp = gp['results']['best']['nodecount']

        if gp['fitness']['minimisation']:
            if (gp['state']['best']['fitness'] < gp['results']['best']['fitness']) or ((stateComp < bestComp) and (gp['state']['best']['fitness'] == gp['results']['best']['fitness'])):
                updateTrainBest = True
        else:
            if (gp['state']['best']['fitness'] > gp['results']['best']['fitness']) or ((stateComp < bestComp) and (gp['state']['best']['fitness'] == gp['results']['best']['fitness'])):
                updateTrainBest = True

        if updateTrainBest:
            gp['results']['best'] = {
                'fitness': gp['state']['best']['fitness'],
                'individual': gp['state']['best']['individual'],
                'returnvalues': gp['state']['best']['returnvalues'],
                'complexity': gp['state']['best']['complexity'],
                'nodecount': gp['state']['best']['nodecount'],
                'foundatgen': gp['state']['count'] - 1,
                'eval_individual': [[tree2evalstr(ind, gp) for ind in ind_list] for ind_list in gp['state']['best']['individual']]
            }

    return gp
