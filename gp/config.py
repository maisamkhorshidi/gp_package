import numpy as np
import os

def configure(gp, xtr1, ytr1, xval1, yval1, xts1, yts1, runname):
    indg = [1, 4]
    gp.config['runcontrol'] = {
        'pop_size': 25,
        'num_class': len(np.unique(np.concatenate((ytr1, yval1, yts1)))),
        'num_pop': len(indg),
        'num_gen': 150,
        'initial_learning_rate': 0.3,
        'lambda_L1': 0.001,
        'lambda_L2': 0.001,
        'tolgrad': 1e-3,
        'epsilon': 1e-8,
        'max_iteration': 500,
        'stallgen': 20,
        'ensemble_method': 'ann',
        'useboruta': False,
        'borutafraction': 1,
        'layer': [8, 16, 8],
        'borutaMaxIter': 10,
        'borutaNoiseN': 2,
        'borutalearner': [8],
        'borutagenes': 5,
        'interaction': [[1, 0], [0, 1]] * len(indg)
    }
    gp.config['selection'] = {
        'tournament_size': [2] * gp.config['runcontrol']['num_pop'],
        'elite_fraction': [0.05] * gp.config['runcontrol']['num_pop'],
        'elite_fraction_ensemble': [0.05] * gp.config['runcontrol']['num_pop']
    }
    gp.config['nodes'] = {
        'const': {
            'p_ERC': [0.1] * gp.config['runcontrol']['num_pop'],
            'p_int': [0] * gp.config['runcontrol']['num_pop'],
            'range': [-1, 1]
        },
        'functions': {
            'name': ['times', 'plus', 'minus']
        }
    }
    gp.config['treedef'] = {
        'max_depth': [10] * gp.config['runcontrol']['num_pop'],
        'max_mutate_depth': [10] * gp.config['runcontrol']['num_pop'],
        'max_nodes': [float('inf')] * gp.config['runcontrol']['num_pop']
    }
    gp.config['fitness'] = {
        'fitfun': gp.EnsembleClassification_fitfun,
        'terminate': True,
        'terminate_value': 0.003
    }
    gp.config['genes'] = {
        'max_genes': [5] * gp.config['runcontrol']['num_pop']
    }

    idx = []
    for i in range(len(indg)):
        if i == 0:
            idst = 0
            iden = xtr1[indg[i]].shape[1]
        else:
            idst = iden
            iden = iden + xtr1[indg[i]].shape[1]
        idx.append(list(range(idst, iden)))
    
    gp.userdata['xindex'] = idx

    xxtr = np.concatenate([xtr1[i] for i in indg], axis=1)
    xxval = np.concatenate([xval1[i] for i in indg], axis=1)
    xxts = np.concatenate([xts1[i] for i in indg], axis=1)

    gp.userdata['xtrain'] = xxtr
    gp.userdata['ytrain'] = ytr1
    gp.userdata['xvalid'] = xxval
    gp.userdata['yvalid'] = yval1
    gp.userdata['xtest'] = xxts
    gp.userdata['ytest'] = yts1
    gp.userdata['ybinarytrain'] = gp.binarymapping(ytr1, len(np.unique(np.concatenate((ytr1, yval1, yts1)))))
    gp.userdata['ybinaryvalid'] = gp.binarymapping(yval1, len(np.unique(np.concatenate((ytr1, yval1, yts1)))))
    gp.userdata['ybinarytest'] = gp.binarymapping(yts1, len(np.unique(np.concatenate((ytr1, yval1, yts1)))))
    gp.userdata['name'] = runname
    gp.userdata['PlotName'] = os.path.join('Results', runname + '.png')
    gp.config['class'] = {
        'userstop': False
    }
