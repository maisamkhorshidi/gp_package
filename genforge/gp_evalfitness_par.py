import numpy as np
from multiprocessing import Pool
from .gp_evaluate_tree import gp_evaluate_tree
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
from .gp_evaluate_softmax import gp_evaluate_softmax
from .gp_evaluate_ensemble import gp_evaluate_ensemble
from .gp_getdepth import gp_getdepth
import copy

def evaluate_individual(args):
    gp, id_pop, id_ind, xtr, xval, xts, function_map = args
    ind = gp.population[id_pop][id_ind]
    gene_out_tr = np.zeros((xtr.shape[0], len(ind)))
    gene_out_val = np.zeros((xval.shape[0], len(ind)))
    gene_out_ts = np.zeros((xts.shape[0], len(ind)))
    num_nodes = np.zeros((1, len(ind)))
    depth = np.zeros((1, len(ind)))
    complexities_isolated = 0
    
    for id_gene in range(len(ind)):
        # Evaluating genes
        gene_out_tr[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xtr, function_map[id_pop]))
        gene_out_val[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xval, function_map[id_pop]))
        gene_out_ts[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xts, function_map[id_pop]))

        depth[id_gene] = gp_getdepth(ind[id_gene])
        num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])
        if gp.config['fitness']['complexityMeasure'] == 1:
            complexities_isolated += gp_getcomplexity(ind[id_gene])
        else:
            complexities_isolated += gp_getnumnodes(ind[id_gene])

    # Training the softmax
    results = gp_evaluate_softmax(gp, id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts)
    
    return id_ind, gene_out_tr, gene_out_val, gene_out_ts, num_nodes, depth, complexities_isolated, results

def gp_evalfitness_par(gp):
    """Evaluate the fitness of individuals (parallel version)."""
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    function_map = gp.config['nodes']['functions']['function']
    xtr = gp.userdata['xtrain']
    xval = gp.userdata['xval']
    xts = gp.userdata['xtest']
    
    pool = Pool(gp.config['runcontrol']['parallel']['n_jobs'])
    for id_pop in range(num_pop):
        args = [(gp, id_pop, id_ind, xtr, xval, xts, function_map) for id_ind in range(pop_size)]
        results = pool.map(evaluate_individual, args)
        
        for result in results:
            id_ind, gene_out_tr, gene_out_val, gene_out_ts, num_nodes, depth, complexities_isolated, results = result
            # Assign the parameters
            prob_tr, prob_val, prob_ts, loss_tr, loss_val, loss_ts, yp_tr, yp_val, yp_ts, weight_genes, fit_tr, fit_val, fit_ts = results
            
            gp.individuals['gene_output']['train'][id_pop][id_ind] = copy.deepcopy(gene_out_tr)
            gp.individuals['gene_output']['validation'][id_pop][id_ind] = copy.deepcopy(gene_out_val)
            gp.individuals['gene_output']['test'][id_pop][id_ind] = copy.deepcopy(gene_out_ts)
            gp.individuals['prob']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(prob_tr)
            gp.individuals['prob']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(prob_val)
            gp.individuals['prob']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(prob_ts)
            gp.individuals['loss']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(loss_tr)
            gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(loss_val)
            gp.individuals['loss']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(loss_ts)
            gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(fit_tr)
            gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(fit_val)
            gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(fit_ts)
            gp.individuals['yp']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(yp_tr)
            gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(yp_val)
            gp.individuals['yp']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(yp_ts)
            gp.individuals['depth']['isolated'][id_pop][id_ind] = copy.deepcopy(depth)
            gp.individuals['num_nodes']['isolated'][id_pop][id_ind] = copy.deepcopy(num_nodes)
            gp.individuals['weight_genes'][id_pop][id_ind] = copy.deepcopy(weight_genes)
            gp.individuals['complexity']['isolated'][id_ind, id_pop] = copy.deepcopy(complexities_isolated)

    pool.close()
    pool.join()
    
    # Evaluating the Ensembles (This part can also be parallelized similarly)
    results_en = gp_evaluate_ensemble(gp)
    
    # Assigning the results
    en_weight = copy.deepcopy(results_en[0])
    en_idx = copy.deepcopy(results_en[1])
    complexity_en = copy.deepcopy(results_en[2])
    prob_en_tr = copy.deepcopy(results_en[3])
    prob_en_val = copy.deepcopy(results_en[4])
    prob_en_ts = copy.deepcopy(results_en[5])
    loss_en_tr = copy.deepcopy(results_en[6])
    loss_en_val = copy.deepcopy(results_en[7])
    loss_en_ts = copy.deepcopy(results_en[8])
    fit_en_tr = copy.deepcopy(results_en[9])
    fit_en_val = copy.deepcopy(results_en[10])
    fit_en_ts = copy.deepcopy(results_en[11])
    yp_en_tr = copy.deepcopy(results_en[12])
    yp_en_val = copy.deepcopy(results_en[13])
    yp_en_ts = copy.deepcopy(results_en[14])
    depth_en = copy.deepcopy(results_en[15])
    num_nodes_en = copy.deepcopy(results_en[16])

    # Assigning the values
    gp.individuals['ensemble_weight'] = copy.deepcopy(en_weight)
    gp.individuals['ensemble_idx'] = copy.deepcopy(en_idx)
    gp.individuals['complexity']['ensemble'] = copy.deepcopy(complexity_en)
    gp.individuals['prob']['ensemble']['train'] = copy.deepcopy(prob_en_tr)
    gp.individuals['prob']['ensemble']['validation'] = copy.deepcopy(prob_en_val)
    gp.individuals['prob']['ensemble']['test'] = copy.deepcopy(prob_en_ts)
    gp.individuals['fitness']['ensemble']['train'] = copy.deepcopy(fit_en_tr)
    gp.individuals['fitness']['ensemble']['validation'] = copy.deepcopy(fit_en_val)
    gp.individuals['fitness']['ensemble']['test'] = copy.deepcopy(fit_en_ts)
    gp.individuals['loss']['ensemble']['train'] = copy.deepcopy(loss_en_tr)
    gp.individuals['loss']['ensemble']['validation'] = copy.deepcopy(loss_en_val)
    gp.individuals['loss']['ensemble']['test'] = copy.deepcopy(loss_en_ts)
    gp.individuals['yp']['ensemble']['train'] = copy.deepcopy(yp_en_tr)
    gp.individuals['yp']['ensemble']['validation'] = copy.deepcopy(yp_en_val)
    gp.individuals['yp']['ensemble']['test'] = copy.deepcopy(yp_en_ts)
    gp.individuals['num_nodes']['ensemble'] = copy.deepcopy(num_nodes_en)
    gp.individuals['depth']['ensemble'] = copy.deepcopy(depth_en)
    
    # Assigning Fitness value
    gp.fitness['values'] = copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
    gp.fitness['complexities'] = copy.deepcopy(gp.individuals['complexity']['ensemble'])
    # Assign the tracking parameters
    gp.track['complexity']['isolated'][gen] = copy.deepcopy(gp.individuals['complexity']['isolated'])
    gp.track['complexity']['ensemble'][gen] = copy.deepcopy(gp.individuals['complexity']['ensemble'])
    gp.track['fitness']['isolated']['train'][gen] = copy.deepcopy(gp.individuals['fitness']['isolated']['train'])
    gp.track['fitness']['isolated']['validation'][gen] = copy.deepcopy(gp.individuals['fitness']['isolated']['validation'])
    gp.track['fitness']['isolated']['test'][gen] = copy.deepcopy(gp.individuals['fitness']['isolated']['test'])
    gp.track['fitness']['ensemble']['train'][gen] = copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
    gp.track['fitness']['ensemble']['validation'][gen] = copy.deepcopy(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['fitness']['ensemble']['test'][gen] = copy.deepcopy(gp.individuals['fitness']['ensemble']['test'])
    gp.track['std_fitness']['isolated']['train'][gen] = np.std(gp.individuals['fitness']['isolated']['train'], axis=0)
    gp.track['std_fitness']['isolated']['validation'][gen] = np.std(gp.individuals['fitness']['isolated']['validation'], axis=0) 
    gp.track['std_fitness']['isolated']['test'][gen] = np.std(gp.individuals['fitness']['isolated']['test'], axis=0) 
    gp.track['std_fitness']['ensemble']['train'][gen] = np.std(gp.individuals['fitness']['ensemble']['train'])
    gp.track['std_fitness']['ensemble']['validation'][gen] = np.std(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['std_fitness']['ensemble']['test'][gen] = np.std(gp.individuals['fitness']['ensemble']['test'])
    gp.track['mean_fitness']['isolated']['train'][gen] = np.mean(gp.individuals['fitness']['isolated']['train'], axis=0)  
    gp.track['mean_fitness']['isolated']['validation'][gen] = np.mean(gp.individuals['fitness']['isolated']['validation'], axis=0) 
    gp.track['mean_fitness']['isolated']['test'][gen] = np.mean(gp.individuals['fitness']['isolated']['test'], axis=0) 
    gp.track['mean_fitness']['ensemble']['train'][gen] = np.mean(gp.individuals['fitness']['ensemble']['train'])
    gp.track['mean_fitness']['ensemble']['validation'][gen] = np.mean(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['mean_fitness']['ensemble']['test'][gen] = np.mean(gp.individuals['fitness']['ensemble']['test'])
    gp.track['ensemble_idx'][gen] = copy.deepcopy(gp.individuals['ensemble_idx'])
    gp.track['depth']['isolated'][gen] = copy.deepcopy(gp.individuals['depth']['isolated'])
    gp.track['depth']['ensemble'][gen] = copy.deepcopy(gp.individuals['depth']['ensemble'])
    gp.track['num_nodes']['isolated'][gen] = copy.deepcopy(gp.individuals['num_nodes']['isolated'])
    gp.track['num_nodes']['ensemble'][gen] = copy.deepcopy(gp.individuals['num_nodes']['ensemble'])
    
    for id_pop in range(num_pop):
        gp.track['rank']['complexity']['isolated'][gen][:, id_pop] = np.argsort(gp.individuals['complexity']['isolated'][:, id_pop]) 
    
    gp.track['rank']['complexity']['ensemble'][gen] = np.argsort(gp.individuals['complexity']['ensemble'])
    
    if gp.config['runcontrol']['minimisation']:
        gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['train'][:, id_pop])
            if xval is not None:
                gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['validation'][:, id_pop])
            if xts is not None:
                gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(gp.individuals['fitness']['isolated']['test'][:, id_pop])
    else:
        gp.track['rank']['fitness']['ensemble']['train'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['train'])
        if xval is not None:
            gp.track['rank']['fitness']['ensemble']['validation'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['validation'])
        if xts is not None:
            gp.track['rank']['fitness']['ensemble']['test'][gen] = np.argsort(-gp.individuals['fitness']['ensemble']['test'])
        
        for id_pop in range(num_pop):
            gp.track['rank']['fitness']['isolated']['train'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['train'][:, id_pop])
            if xval is not None:
                gp.track['rank']['fitness']['isolated']['validation'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['validation'][:, id_pop])
            if xts is not None:
                gp.track['rank']['fitness']['isolated']['test'][gen][id_pop] = np.argsort(-gp.individuals['fitness']['isolated']['test'][:, id_pop])

