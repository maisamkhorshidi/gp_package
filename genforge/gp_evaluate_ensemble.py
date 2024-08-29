import numpy as np
from itertools import product
from scipy.optimize import minimize
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import copy

def gp_evaluate_ensemble(gp):
    """
    Calculate the best ensemble weights for each individual in the id_pop=0 population by combining them
    with every combination of individuals from other populations.

    Args:
    gp (object): The GP object containing individuals' probabilities and labels.

    Returns:
    None
    """
    pop_size = gp.config['runcontrol']['pop_size']
    num_pop = gp.config['runcontrol']['num_pop']
    depth = copy.deepcopy(gp.individuals['depth']['isolated'])
    num_nodes = copy.deepcopy(gp.individuals['num_nodes']['isolated'])
    
    id_pop = 0
    # num_class = gp.config['runcontrol']['num_class']
    num_individuals_pop1 = len(gp.individuals['prob']['isolated']['train'][id_pop])
    
    # Extract the training probabilities and labels
    prob_train_pop1 = copy.deepcopy(gp.individuals['prob']['isolated']['train'][id_pop])
    y_train = gp.userdata['ytrain']
    y_val = gp.userdata['yval']
    y_test = gp.userdata['ytest']
    
    # Prepare lists to hold the best results
    best_ensemble_prob = [None for _ in range(num_individuals_pop1)]
    best_weights = [None for _ in range(num_individuals_pop1)]
    best_loss = np.full(num_individuals_pop1, np.inf)
    best_idx = [None for _ in range(num_individuals_pop1)]  # To store the best ensemble combination indices
    
    # Iterate through each individual in the first population
    for id_ind in range(num_individuals_pop1):
        prob_train_ensemble = [prob_train_pop1[id_ind]]  # Start with the individual from the first population
        
        # Iterate through all possible combinations of individuals from the other populations
        other_populations = [p for p in range(1, len(gp.individuals['prob']['isolated']['train']))]
        combinations = list(product(*[range(len(gp.individuals['prob']['isolated']['train'][p])) for p in other_populations]))
        
        # ii = -1
        for combo in combinations:
            # ii += 1
            prob_combo = copy.deepcopy(prob_train_ensemble)
            
            for j, pop_id in enumerate(other_populations):
                prob_combo.append(copy.deepcopy(gp.individuals['prob']['isolated']['train'][pop_id][combo[j]]))
            
            # Define the loss function for optimizing ensemble weights
            def loss_function(weights):
                combined_prob = np.zeros_like(prob_combo[0])
                # bias = weights[-num_class:]
                for k in range(len(prob_combo)):
                    combined_prob += weights[k] * prob_combo[k]
                # combined_prob += bias
                combined_prob = np.clip(combined_prob, 1e-15, 1 - 1e-15)  # Avoid log(0) error
                loss = SparseCategoricalCrossentropy()(y_train, combined_prob)
                return loss.numpy()

            # Initial guess for weights (even distribution) and biases (zero)
            initial_weights = np.ones(len(prob_combo)) / len(prob_combo)
            # initial_biases = np.zeros(num_class)
            # initial_params = np.concatenate([initial_weights, initial_biases])
            
            # Constraints: Weights must sum to 1
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w[:len(prob_combo)]) - 1}]
            
            # Bounds: Weights and biases can be any real number
            bounds = [(None, None) for _ in range(len(prob_combo))] #+ [(None, None)] * num_class

            
            # print('a: ', id_ind, 'combo: ', ii, ' of ', len(combinations))
            # Optimize the weights and biases
            # result = minimize(loss_function, initial_params, bounds=bounds, constraints=constraints)
            result = minimize(loss_function, initial_weights, bounds=bounds, constraints=constraints)

            # If this combination has the best loss, store it
            if result.fun < best_loss[id_ind]:
                best_loss[id_ind] = copy.deepcopy(result.fun)
                best_weights[id_ind] = copy.deepcopy(result.x)
                best_ensemble_prob[id_ind] = np.zeros_like(prob_combo[0])
                
                for k in range(len(prob_combo)):
                    best_ensemble_prob[id_ind] += result.x[k] * prob_combo[k]
                # best_ensemble_prob[id_ind] += result.x[-num_class:]
                best_idx[id_ind] = [id_ind] + list(combo)  # Store the indices of the best combination

    # Compute ensemble complexity
    complexity_ensemble = np.zeros((pop_size))
    for id_ind in range(pop_size):
        for pop_id in range(len(best_idx[id_ind])):
            complexity_ensemble[id_ind] += gp.individuals['complexity']['isolated'][best_idx[id_ind][pop_id], pop_id]
    
    # Assign the best results 
    en_weight = [None for _ in range(pop_size)]
    en_idx = np.zeros((pop_size, num_pop))
    complexity_en = np.zeros((pop_size))
    prob_en_tr = [None for _ in range(pop_size)]
    prob_en_val = [None for _ in range(pop_size)]
    prob_en_ts = [None for _ in range(pop_size)]
    loss_en_tr = np.zeros((pop_size))
    loss_en_val = np.zeros((pop_size))
    loss_en_ts = np.zeros((pop_size))
    fit_en_tr = np.full(pop_size, np.inf)
    fit_en_val = np.full(pop_size, np.inf)
    fit_en_ts = np.full(pop_size, np.inf)
    yp_en_tr = [None for _ in range(pop_size)]
    yp_en_val = [None for _ in range(pop_size)]
    yp_en_ts = [None for _ in range(pop_size)]
    depth_en = [None for _ in range(pop_size)]
    num_nodes_en = [None for _ in range(pop_size)]
    
    for id_ind in range(num_individuals_pop1):
        en_weight[id_ind] =         copy.deepcopy(best_weights[id_ind])
        en_idx[id_ind,:] =          copy.deepcopy(best_idx[id_ind])
        complexity_en[id_ind] =     copy.deepcopy(complexity_ensemble[id_ind])
        prob_en_tr[id_ind] =        copy.deepcopy(best_ensemble_prob[id_ind])
        loss_en_tr[id_ind] =        copy.deepcopy(best_loss[id_ind])
        fit_en_tr[id_ind] =         copy.deepcopy(best_loss[id_ind])
        yp_en_tr[id_ind] =          np.argmax(best_ensemble_prob[id_ind], axis=1)
        depth_en[id_ind] =          list()
        num_nodes_en[id_ind] =      list()
        for id_pop in range(num_pop):
            num_nodes_en[id_ind].append(copy.deepcopy(num_nodes[id_pop][int(en_idx[id_ind,id_pop])]))
            depth_en[id_ind].append(copy.deepcopy(depth[id_pop][int(en_idx[id_ind,id_pop])]))

        # Repeat for validation and test sets if available
        if y_val is not None:
            prob_val_combo = [copy.deepcopy(gp.individuals['prob']['isolated']['validation'][id_pop][int(best_idx[id_ind][0])])]
            for j, pop_id in enumerate(other_populations):
                prob_val_combo.append(copy.deepcopy(gp.individuals['prob']['isolated']['validation'][pop_id][int(best_idx[id_ind][j+1])]))
            combined_prob_val = np.zeros_like(prob_val_combo[0])
            for k in range(len(prob_val_combo)):
                combined_prob_val += best_weights[id_ind][k] * prob_val_combo[k]
            
            prob_en_val[id_ind] =   copy.deepcopy(combined_prob_val)
            fit_en_val[id_ind] =    SparseCategoricalCrossentropy()(y_val, combined_prob_val).numpy()
            loss_en_val[id_ind] =   copy.deepcopy(fit_en_val[id_ind])
            yp_en_val[id_ind] =     np.argmax(combined_prob_val, axis=1)
            
        if y_test is not None:
            prob_ts_combo = [gp.individuals['prob']['isolated']['test'][id_pop][int(best_idx[id_ind][0])]]
            for j, pop_id in enumerate(other_populations):
                prob_ts_combo.append(gp.individuals['prob']['isolated']['test'][pop_id][int(best_idx[id_ind][j+1])])
            combined_prob_ts = np.zeros_like(prob_ts_combo[0])
            for k in range(len(prob_ts_combo)):
                combined_prob_ts += best_weights[id_ind][k] * prob_ts_combo[k]
            
            prob_en_ts[id_ind] =    copy.deepcopy(combined_prob_ts)
            fit_en_ts[id_ind] =     SparseCategoricalCrossentropy()(gp.userdata['ytest'], combined_prob_ts).numpy()
            loss_en_ts[id_ind] =    copy.deepcopy(fit_en_ts[id_ind])
            yp_en_ts[id_ind] =      np.argmax(combined_prob_ts, axis=1).copy()
            
    # Store in the results list
    results_en = [
        en_weight,
        en_idx,
        complexity_en,
        prob_en_tr,
        prob_en_val,
        prob_en_ts,
        loss_en_tr,
        loss_en_val,
        loss_en_ts,
        fit_en_tr,
        fit_en_val,
        fit_en_ts,
        yp_en_tr,
        yp_en_val,
        yp_en_ts,
        depth_en,
        num_nodes_en,
    ]
    return results_en
            
            
            