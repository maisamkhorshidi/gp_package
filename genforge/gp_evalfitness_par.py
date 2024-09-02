import numpy as np
from multiprocessing import Pool, shared_memory
from .gp_evaluate_tree import gp_evaluate_tree
from .gp_getcomplexity import gp_getcomplexity
from .gp_getnumnodes import gp_getnumnodes
from .gp_evaluate_softmax import gp_evaluate_softmax
from .gp_evaluate_ensemble import gp_evaluate_ensemble
from .gp_getdepth import gp_getdepth
import copy

def create_shared_memory(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shm_arr, arr)
    return shm, shm_arr

def evaluate_individual(args):
    id_pop, id_ind, shm_names, shapes, dtypes, function_map, ind, complexity_measure, num_class,\
        learning_rate, optimizer_type, initializer, regularization, regularization_rate,\
        batch_size, epochs, momentum, decay, clipnorm, clipvalue, patience = args
    # Access the shared memory arrays
    shm_xtr = shared_memory.SharedMemory(name=shm_names['xtr'])
    shm_xval = shared_memory.SharedMemory(name=shm_names['xval'])
    shm_xts = shared_memory.SharedMemory(name=shm_names['xts'])
    shm_ytr = shared_memory.SharedMemory(name=shm_names['ytr'])
    shm_yval = shared_memory.SharedMemory(name=shm_names['yval'])
    shm_yts = shared_memory.SharedMemory(name=shm_names['yts'])

    xtr = np.ndarray(shapes['xtr'], dtype=dtypes['xtr'], buffer=shm_xtr.buf)
    xval = np.ndarray(shapes['xval'], dtype=dtypes['xval'], buffer=shm_xval.buf)
    xts = np.ndarray(shapes['xts'], dtype=dtypes['xts'], buffer=shm_xts.buf)
    ytr = np.ndarray(shapes['ytr'], dtype=dtypes['ytr'], buffer=shm_ytr.buf)
    yval = np.ndarray(shapes['yval'], dtype=dtypes['yval'], buffer=shm_yval.buf)
    yts = np.ndarray(shapes['yts'], dtype=dtypes['yts'], buffer=shm_yts.buf)

    # Perform the fitness evaluation
    gene_out_tr = np.zeros((xtr.shape[0], len(ind)))
    gene_out_val = np.zeros((xval.shape[0], len(ind)))
    gene_out_ts = np.zeros((xts.shape[0], len(ind)))
    num_nodes = np.zeros((len(ind)))
    depth = np.zeros((len(ind)))
    complexities_isolated = 0

    for id_gene in range(len(ind)):
        # Evaluating genes
        gene_out_tr[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xtr, function_map[id_pop]))
        gene_out_val[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xval, function_map[id_pop]))
        gene_out_ts[:, id_gene] = np.tanh(gp_evaluate_tree(ind[id_gene], xts, function_map[id_pop]))
        depth[id_gene] = gp_getdepth(ind[id_gene])
        num_nodes[id_gene] = gp_getnumnodes(ind[id_gene])
        if complexity_measure == 1:
            complexities_isolated += gp_getcomplexity(ind[id_gene])
        else:
            complexities_isolated += gp_getnumnodes(ind[id_gene])

    args = ytr, yval, yts, num_class, learning_rate[id_pop], optimizer_type, initializer,\
        regularization, regularization_rate, batch_size, epochs, momentum,\
            decay, clipnorm, clipvalue, patience,\
                id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts

    # Training the softmax
    results = gp_evaluate_softmax(args)

    return (id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts, depth, num_nodes, complexities_isolated, *results)

def gp_evalfitness_par(gp):
    """Evaluate the fitness of individuals (parallel version)."""
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    function_map = gp.config['nodes']['functions']['function']
    popgp = copy.deepcopy(gp.population)
    learning_rate = gp.config['softmax']['learning_rate']
    optimizer_type = gp.config['softmax']['optimizer_type']
    initializer = gp.config['softmax']['initializer']
    regularization = gp.config['softmax']['regularization']
    regularization_rate = gp.config['softmax']['regularization_rate']
    batch_size = gp.config['softmax']['batch_size']
    epochs = gp.config['softmax']['epochs']
    momentum = gp.config['softmax']['momentum']
    decay = gp.config['softmax']['decay']
    clipnorm = gp.config['softmax']['clipnorm']
    clipvalue = gp.config['softmax']['learning_rate']
    patience = gp.config['softmax']['patience']
    complexity_measure = gp.config['fitness']['complexityMeasure']
    num_class = gp.config['runcontrol']['num_class']
    xtr = gp.userdata['xtrain']
    xval = gp.userdata['xval']
    xts = gp.userdata['xtest']
    ytr = gp.userdata['ytrain']
    yval = gp.userdata['yval']
    yts = gp.userdata['ytest']
    
    # Create shared memory for large arrays
    shm_xtr, xtr_shared = create_shared_memory(xtr)
    shm_xval, xval_shared = create_shared_memory(xval)
    shm_xts, xts_shared = create_shared_memory(xts)
    shm_ytr, ytr_shared = create_shared_memory(ytr)
    shm_yval, yval_shared = create_shared_memory(yval)
    shm_yts, yts_shared = create_shared_memory(yts)

    # Shared memory names, shapes, and dtypes
    shm_names = {
        'xtr': shm_xtr.name,
        'xval': shm_xval.name,
        'xts': shm_xts.name,
        'ytr': shm_ytr.name,
        'yval': shm_yval.name,
        'yts': shm_yts.name
    }
    shapes = {
        'xtr': xtr_shared.shape,
        'xval': xval_shared.shape,
        'xts': xts_shared.shape,
        'ytr': ytr_shared.shape,
        'yval': yval_shared.shape,
        'yts': yts_shared.shape
    }
    dtypes = {
        'xtr': xtr_shared.dtype,
        'xval': xval_shared.dtype,
        'xts': xts_shared.dtype,
        'ytr': ytr_shared.dtype,
        'yval': yval_shared.dtype,
        'yts': yts_shared.dtype
    }

    pool = Pool(gp.config['runcontrol']['parallel']['n_jobs'])
    args = [(id_pop, id_ind, shm_names, shapes, dtypes, function_map, popgp[id_pop][id_ind], complexity_measure, num_class,\
        learning_rate[id_pop], optimizer_type[id_pop], initializer[id_pop], regularization[id_pop], regularization_rate[id_pop],\
        batch_size[id_pop], epochs[id_pop], momentum[id_pop], decay[id_pop], clipnorm[id_pop], clipvalue[id_pop], patience[id_pop]) 
            for id_pop in range(num_pop) 
            for id_ind in range(pop_size)]
    results = pool.starmap(evaluate_individual, args)

    pool.close()
    pool.join()

    # Handle the results
    for result in results:
        id_pop, id_ind, gene_out_tr, gene_out_val, gene_out_ts, depth, num_nodes, complexities_isolated, *fitness_results = result
        # Assign the results back to the gp object
        gp.individuals['gene_output']['train'][id_pop][id_ind] = copy.deepcopy(gene_out_tr)
        gp.individuals['gene_output']['validation'][id_pop][id_ind] = copy.deepcopy(gene_out_val)
        gp.individuals['gene_output']['test'][id_pop][id_ind] = copy.deepcopy(gene_out_ts)
        gp.individuals['depth']['isolated'][id_pop][id_ind] = copy.deepcopy(depth)
        gp.individuals['num_nodes']['isolated'][id_pop][id_ind] = copy.deepcopy(num_nodes)
        gp.individuals['complexity']['isolated'][id_ind, id_pop] = copy.deepcopy(complexities_isolated)

        # Assign fitness results (prob_tr, prob_val, prob_ts, loss_tr, loss_val, loss_ts, yp_tr, yp_val, yp_ts, weight_genes)
        gp.individuals['prob']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(fitness_results[0])
        gp.individuals['prob']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(fitness_results[1])
        gp.individuals['prob']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(fitness_results[2])
        gp.individuals['loss']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(fitness_results[3])
        gp.individuals['loss']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(fitness_results[4])
        gp.individuals['loss']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(fitness_results[5])
        gp.individuals['fitness']['isolated']['train'][id_ind, id_pop] = copy.deepcopy(fitness_results[3])
        gp.individuals['fitness']['isolated']['validation'][id_ind, id_pop] = copy.deepcopy(fitness_results[4])
        gp.individuals['fitness']['isolated']['test'][id_ind, id_pop] = copy.deepcopy(fitness_results[5])
        gp.individuals['yp']['isolated']['train'][id_pop][id_ind] = copy.deepcopy(fitness_results[6])
        gp.individuals['yp']['isolated']['validation'][id_pop][id_ind] = copy.deepcopy(fitness_results[7])
        gp.individuals['yp']['isolated']['test'][id_pop][id_ind] = copy.deepcopy(fitness_results[8])
        gp.individuals['weight_genes'][id_pop][id_ind] = copy.deepcopy(fitness_results[9])

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



    # Cleanup shared memory
    shm_xtr.close()
    shm_xtr.unlink()

    shm_xval.close()
    shm_xval.unlink()

    shm_xts.close()
    shm_xts.unlink()

    shm_ytr.close()
    shm_ytr.unlink()

    shm_yval.close()
    shm_yval.unlink()

    shm_yts.close()
    shm_yts.unlink()
