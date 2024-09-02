import numpy as np
from .gp_evaluate_tree import gp_evaluate_tree
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
from .gp_evaluate_softmax import gp_evaluate_softmax
from .gp_evaluate_ensemble import gp_evaluate_ensemble
from .gp_getdepth import gp_getdepth
import copy
#from .utils import tree2evalstr, getcomplexity, getnumnodes

def gp_evalfitness_ord(gp):
    """Evaluate the fitness of individuals (ordinary version)."""
    gen = gp.state['generation']
    # num_class = gp.userdata['num_class']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    popgp = copy.deepcopy(gp.population)
    complexity_measure = gp.config['fitness']['complexityMeasure']
    function_map = gp.config['nodes']['functions']['function']
    # fitfun = gp.config['fitness']['fitfun']
    # usecache = gp.config['runcontrol']['usecache']
    xtr = gp.userdata['xtrain']
    xval = gp.userdata['xval']
    xts = gp.userdata['xtest']
    
    ytr = gp.userdata['ytrain']
    yval = gp.userdata['yval']
    yts = gp.userdata['ytest']
    num_class = gp.config['runcontrol']['num_class']
    

    for id_pop in range(num_pop):
        pop = popgp[id_pop]
        # softmax parameters
        learning_rate = gp.config['softmax']['learning_rate'][id_pop]
        optimizer_type = gp.config['softmax']['optimizer_type'][id_pop]
        initializer = gp.config['softmax']['initializer'][id_pop]
        regularization = gp.config['softmax']['regularization'][id_pop]
        regularization_rate = gp.config['softmax']['regularization_rate'][id_pop]
        batch_size = gp.config['softmax']['batch_size'][id_pop]
        epochs = gp.config['softmax']['epochs'][id_pop]
        momentum = gp.config['softmax']['momentum'][id_pop]
        decay = gp.config['softmax']['decay'][id_pop]
        clipnorm = gp.config['softmax']['clipnorm'][id_pop]
        clipvalue = gp.config['softmax']['learning_rate'][id_pop]
        patience = gp.config['softmax']['patience'][id_pop]
        # Update state to index of the individual that is about to be evaluated
        for id_ind in range(pop_size):
            # print(' ')
            # print('generation:', gen)
            # print('id_pop:',id_pop)
            # print('id_ind:',id_ind)
            if gp.config['runcontrol']['usecache'] and gp.cache['gene_output']['train'][id_pop][id_ind] is not None:
                gene_out_tr =               copy.deepcopy(gp.cache['gene_output']['train'][id_pop][id_ind])
                gene_out_val =              copy.deepcopy(gp.cache['gene_output']['validation'][id_pop][id_ind])
                gene_out_ts =               copy.deepcopy(gp.cache['gene_output']['test'][id_pop][id_ind])
                prob_tr =                   copy.deepcopy(gp.cache['prob']['isolated']['train'][id_pop][id_ind])
                prob_val =                  copy.deepcopy(gp.cache['prob']['isolated']['validation'][id_pop][id_ind])
                prob_ts =                   copy.deepcopy(gp.cache['prob']['isolated']['test'][id_pop][id_ind])
                loss_tr =                   copy.deepcopy(gp.cache['loss']['isolated']['train'][id_pop][id_ind])
                loss_val =                  copy.deepcopy(gp.cache['loss']['isolated']['validation'][id_pop][id_ind])
                loss_ts =                   copy.deepcopy(gp.cache['loss']['isolated']['test'][id_pop][id_ind])
                yp_tr =                     copy.deepcopy(gp.cache['yp']['isolated']['train'][id_pop][id_ind])
                yp_val =                    copy.deepcopy(gp.cache['yp']['isolated']['validation'][id_pop][id_ind])
                yp_ts =                     copy.deepcopy(gp.cache['yp']['isolated']['test'][id_pop][id_ind])
                fit_tr =                    copy.deepcopy(gp.cache['fitness']['isolated']['train'][id_pop][id_ind])
                fit_val =                   copy.deepcopy(gp.cache['fitness']['isolated']['validation'][id_pop][id_ind])
                fit_ts =                    copy.deepcopy(gp.cache['fitness']['isolated']['test'][id_pop][id_ind])
                depth =                     copy.deepcopy(gp.cache['depth']['isolated'][id_pop][id_ind])
                num_nodes =                 copy.deepcopy(gp.cache['num_nodes']['isolated'][id_pop][id_ind])
                weight_genes =              copy.deepcopy(gp.cache['weight_genes'][id_pop][id_ind])
                complexities_isolated =     copy.deepcopy(gp.cache['complexity']['isolated'][id_pop][id_ind])
            else:
                ind = pop[id_ind]
                gene_out_tr = np.zeros((xtr.shape[0], len(ind)))
                gene_out_val = np.zeros((xval.shape[0], len(ind)))
                gene_out_ts = np.zeros((xts.shape[0], len(ind)))
                num_nodes = np.zeros((len(ind)))
                depth = np.zeros((len(ind)))
                complexities_isolated = 0
                for id_gene in range(len(ind)):
                    # Evaluating genes
                    gene_out_tr [:,id_gene] = np.tanh(gp_evaluate_tree(ind [id_gene], xtr, function_map[id_pop]))
                    gene_out_val [:,id_gene] = np.tanh(gp_evaluate_tree(ind [id_gene], xval, function_map[id_pop]))
                    gene_out_ts [:,id_gene] = np.tanh(gp_evaluate_tree(ind [id_gene], xts, function_map[id_pop]))
                    
                    depth[id_gene] = gp_getdepth(ind[id_gene])
                    num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])
                    if complexity_measure == 1:
                        complexities_isolated += gp_getcomplexity(ind[id_gene])
                    else:
                        complexities_isolated += gp_getnumnodes(ind[id_gene])
                    
                args = ytr, yval, yts, num_class, learning_rate, optimizer_type, initializer,\
                    regularization, regularization_rate, batch_size, epochs, momentum, decay,\
                        clipnorm, clipvalue, patience, id_pop, id_ind, gene_out_tr, gene_out_val,\
                            gene_out_ts
                # Training the softmax
                results = gp_evaluate_softmax(args)
                # Assign results
                prob_tr =       results[0]
                prob_val =      results[1]
                prob_ts =       results[2]
                loss_tr =       results[3]
                loss_val =      results[4]
                loss_ts =       results[5]
                yp_tr =         results[6]
                yp_val =        results[7]
                yp_ts =         results[8]
                weight_genes =  results[9]
                fit_tr =        results[3]
                fit_val =       results[4]
                fit_ts =        results[5]
                
    
            # Assign the parameters
            gp.individuals['gene_output']['train'][id_pop][id_ind] =                copy.deepcopy(gene_out_tr)
            gp.individuals['gene_output']['validation'][id_pop][id_ind] =           copy.deepcopy(gene_out_val)
            gp.individuals['gene_output']['test'][id_pop][id_ind] =                 copy.deepcopy(gene_out_ts)
            gp.individuals['prob']['isolated']['train'][id_pop][id_ind] =           copy.deepcopy(prob_tr)
            gp.individuals['prob']['isolated']['validation'][id_pop][id_ind] =      copy.deepcopy(prob_val)
            gp.individuals['prob']['isolated']['test'][id_pop][id_ind] =            copy.deepcopy(prob_ts)
            gp.individuals['loss']['isolated']['train'][id_ind, id_pop] =           copy.deepcopy(loss_tr)
            gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] =      copy.deepcopy(loss_val)
            gp.individuals['loss']['isolated']['test'][id_ind, id_pop] =            copy.deepcopy(loss_ts)
            gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] =        copy.deepcopy(fit_tr)
            gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] =   copy.deepcopy(fit_val)
            gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] =         copy.deepcopy(fit_ts)
            gp.individuals['yp']['isolated']['train'][id_pop][id_ind] =             copy.deepcopy(yp_tr)
            gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] =        copy.deepcopy(yp_val)
            gp.individuals['yp']['isolated']['test'][id_pop][id_ind] =              copy.deepcopy(yp_ts)
            gp.individuals['depth']['isolated'][id_pop][id_ind] =                   copy.deepcopy(depth)
            gp.individuals['num_nodes']['isolated'][id_pop][id_ind] =               copy.deepcopy(num_nodes)
            gp.individuals['weight_genes'][id_pop][id_ind] =                        copy.deepcopy(weight_genes) 
            gp.individuals['complexity']['isolated'][id_ind, id_pop] =              copy.deepcopy(complexities_isolated)
            
    if num_pop > 1:
        # Evaluating the Ensembles
        results_en = gp_evaluate_ensemble(gp)
    else:
        results_en = [None, 
                      None,
                      copy.deepcopy(gp.individuals['complexity']['isolated']),
                      copy.deepcopy(gp.individuals['prob']['isolated']['train']),
                      copy.deepcopy(gp.individuals['prob']['isolated']['validation']),
                      copy.deepcopy(gp.individuals['prob']['isolated']['test']),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['train']),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['validation']),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['test']),
                      copy.deepcopy(gp.individuals['loss']['isolated']['train']),
                      copy.deepcopy(gp.individuals['loss']['isolated']['validation']),
                      copy.deepcopy(gp.individuals['loss']['isolated']['test']),
                      copy.deepcopy(gp.individuals['yp']['isolated']['train']),
                      copy.deepcopy(gp.individuals['yp']['isolated']['validation']),
                      copy.deepcopy(gp.individuals['yp']['isolated']['test']),
                      copy.deepcopy(gp.individuals['depth']['isolated']),
                      copy.deepcopy(gp.individuals['num_nodes']['isolated']),
                      np.arange(0, pop_size),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['train']),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['validation']),
                      copy.deepcopy(gp.individuals['fitness']['isolated']['test']),
                      ]
    
    # Assigning the results
    en_weight =         copy.deepcopy(results_en[0])
    en_idx =            copy.deepcopy(results_en[1])
    complexity_en =     copy.deepcopy(results_en[2])
    prob_en_tr =        copy.deepcopy(results_en[3])
    prob_en_val =       copy.deepcopy(results_en[4])
    prob_en_ts =        copy.deepcopy(results_en[5])
    loss_en_tr =        copy.deepcopy(results_en[6])
    loss_en_val =       copy.deepcopy(results_en[7])
    loss_en_ts =        copy.deepcopy(results_en[8])
    fit_en_tr =         copy.deepcopy(results_en[9])
    fit_en_val =        copy.deepcopy(results_en[10])
    fit_en_ts =         copy.deepcopy(results_en[11])
    yp_en_tr =          copy.deepcopy(results_en[12])
    yp_en_val =         copy.deepcopy(results_en[13])
    yp_en_ts =          copy.deepcopy(results_en[14])
    depth_en =          copy.deepcopy(results_en[15])
    num_nodes_en =      copy.deepcopy(results_en[16])
    id_ens =            copy.deepcopy(results_en[17])
    fit_ens_tr =        copy.deepcopy(results_en[18])
    fit_ens_val =       copy.deepcopy(results_en[19])
    fit_ens_ts =        copy.deepcopy(results_en[20])
    
    # Assigning the values
    gp.individuals['ensemble_weight'] =                         copy.deepcopy(en_weight)
    gp.individuals['ensemble_idx'] =                            copy.deepcopy(en_idx)
    gp.individuals['complexity']['ensemble'] =                  copy.deepcopy(complexity_en)
    gp.individuals['prob']['ensemble']['train'] =               copy.deepcopy(prob_en_tr)
    gp.individuals['prob']['ensemble']['validation'] =          copy.deepcopy(prob_en_val)
    gp.individuals['prob']['ensemble']['test'] =                copy.deepcopy(prob_en_ts)
    gp.individuals['fitness']['ensemble']['train'] =            copy.deepcopy(fit_en_tr)
    gp.individuals['fitness']['ensemble']['validation'] =       copy.deepcopy(fit_en_val)
    gp.individuals['fitness']['ensemble']['test'] =             copy.deepcopy(fit_en_ts)
    gp.individuals['loss']['ensemble']['train'] =               copy.deepcopy(loss_en_tr)
    gp.individuals['loss']['ensemble']['validation'] =          copy.deepcopy(loss_en_val)
    gp.individuals['loss']['ensemble']['test'] =                copy.deepcopy(loss_en_ts)
    gp.individuals['yp']['ensemble']['train'] =                 copy.deepcopy(yp_en_tr)
    gp.individuals['yp']['ensemble']['validation'] =            copy.deepcopy(yp_en_val)
    gp.individuals['yp']['ensemble']['test'] =                  copy.deepcopy(yp_en_ts)
    gp.individuals['num_nodes']['ensemble'] =                   copy.deepcopy(num_nodes_en)
    gp.individuals['depth']['ensemble'] =                       copy.deepcopy(depth_en)
    
    # Assigning Fitness value
    gp.fitness['values'] = copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
    gp.fitness['complexities'] = copy.deepcopy(gp.individuals['complexity']['ensemble'])
    # Assign the tracking parameters
    gp.track['complexity']['isolated'][gen] =                   copy.deepcopy(gp.individuals['complexity']['isolated'])
    gp.track['complexity']['ensemble'][gen] =                   copy.deepcopy(gp.individuals['complexity']['ensemble'])
    gp.track['fitness']['isolated']['train'][gen] =             copy.deepcopy(gp.individuals['fitness']['isolated']['train'])
    gp.track['fitness']['isolated']['validation'][gen] =        copy.deepcopy(gp.individuals['fitness']['isolated']['validation'])
    gp.track['fitness']['isolated']['test'][gen] =              copy.deepcopy(gp.individuals['fitness']['isolated']['test'])
    gp.track['fitness']['ensemble']['train'][gen] =             copy.deepcopy(gp.individuals['fitness']['ensemble']['train'])
    gp.track['fitness']['ensemble']['validation'][gen] =        copy.deepcopy(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['fitness']['ensemble']['test'][gen] =              copy.deepcopy(gp.individuals['fitness']['ensemble']['test'])
    gp.track['std_fitness']['isolated']['train'][gen] =         np.std(gp.individuals['fitness']['isolated']['train'], axis = 0)
    gp.track['std_fitness']['isolated']['validation'][gen] =    np.std(gp.individuals['fitness']['isolated']['validation'], axis = 0) 
    gp.track['std_fitness']['isolated']['test'][gen] =          np.std(gp.individuals['fitness']['isolated']['test'], axis = 0) 
    gp.track['std_fitness']['ensemble']['train'][gen] =         np.std(gp.individuals['fitness']['ensemble']['train'])
    gp.track['std_fitness']['ensemble']['validation'][gen] =    np.std(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['std_fitness']['ensemble']['test'][gen] =          np.std(gp.individuals['fitness']['ensemble']['test'])
    gp.track['mean_fitness']['isolated']['train'][gen] =        np.mean(gp.individuals['fitness']['isolated']['train'], axis = 0)  
    gp.track['mean_fitness']['isolated']['validation'][gen] =   np.mean(gp.individuals['fitness']['isolated']['validation'], axis = 0) 
    gp.track['mean_fitness']['isolated']['test'][gen] =         np.mean(gp.individuals['fitness']['isolated']['test'], axis = 0) 
    gp.track['mean_fitness']['ensemble']['train'][gen] =        np.mean(gp.individuals['fitness']['ensemble']['train'])
    gp.track['mean_fitness']['ensemble']['validation'][gen] =   np.mean(gp.individuals['fitness']['ensemble']['validation'])
    gp.track['mean_fitness']['ensemble']['test'][gen] =         np.mean(gp.individuals['fitness']['ensemble']['test'])
    gp.track['ensemble_idx'][gen] =                             copy.deepcopy(gp.individuals['ensemble_idx'])
    gp.track['depth']['isolated'][gen] =                        copy.deepcopy(gp.individuals['depth']['isolated'])
    gp.track['depth']['ensemble'][gen] =                        copy.deepcopy(gp.individuals['depth']['ensemble'])
    gp.track['num_nodes']['isolated'][gen] =                    copy.deepcopy(gp.individuals['num_nodes']['isolated'])
    gp.track['num_nodes']['ensemble'][gen] =                    copy.deepcopy(gp.individuals['num_nodes']['ensemble'])
    gp.track['all_ensemble']['idx'][gen] =                      copy.deepcopy(id_ens)
    gp.track['all_ensemble']['fitness']['train'][gen] =         copy.deepcopy(fit_ens_tr)
    gp.track['all_ensemble']['fitness']['validation'][gen] =    copy.deepcopy(fit_ens_val)
    gp.track['all_ensemble']['fitness']['test'][gen] =          copy.deepcopy(fit_ens_ts)               
    
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

