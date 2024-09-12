import numpy as np
from itertools import product
from scipy.optimize import minimize
import copy

def loss(prob, Y):
    log_likelihood = -np.sum(Y * np.log(prob + 1e-9), axis=1)
    loss = np.mean(log_likelihood)
    return loss

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
    ybin_tr = gp.userdata['ybinarytrain']
    ybin_val = gp.userdata['ybinaryval']
    ybin_ts = gp.userdata['ybinarytest']
    
    # Prepare lists to hold the best results
    best_ensemble_prob = [None for _ in range(num_individuals_pop1)]
    best_weights = [None for _ in range(num_individuals_pop1)]
    best_loss = np.full(num_individuals_pop1, np.inf)
    best_idx = [None for _ in range(num_individuals_pop1)]  # To store the best ensemble combination indices
    
    id_ens = np.zeros((pop_size**num_pop, num_pop), dtype = int)
    fit_ens_tr = np.full((pop_size**num_pop), np.inf)
    fit_ens_val = np.full((pop_size**num_pop), np.inf)
    fit_ens_ts = np.full((pop_size**num_pop), np.inf)
    weights = [None for _ in range(pop_size**num_pop)]
    counter = 0
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
            
            def normal_weight(weights1):
                normalized_weights = np.zeros_like(weights1)
                for j in range(prob_combo[0].shape[1]):  # Iterate over classes
                    start_idx = j * len(prob_combo)
                    end_idx = (j + 1) * len(prob_combo)
                    # Normalize weights for each class
                    # print('')
                    # print(weights1)
                    class_weights = weights1[start_idx:end_idx]
                    sum_class_weights = np.sum(class_weights)
                    if sum_class_weights != 0:
                        normalized_weights[start_idx:end_idx] = class_weights / sum_class_weights
                    else:
                        normalized_weights[start_idx:end_idx] = np.zeros_like(class_weights)
                return normalized_weights
            
            def loss_function_other(prob_combo_tr, y_tr, weights1):
                combined_prob = np.zeros_like(prob_combo_tr[0])

                # Normalize weights for each class so that their sum equals 1
                normalized_weights = normal_weight(weights1)
                
                for j in range(prob_combo_tr[0].shape[1]):  # Iterate over classes
                    for k in range(len(prob_combo_tr)):  # Iterate over models
                        combined_prob[:, j] += prob_combo_tr[k][:, j] * normalized_weights[j * len(prob_combo_tr) + k]
                
                combined_prob = np.clip(combined_prob, 1e-12, 1 - 1e-12)  # Avoid log(0) error
                loss1 = loss(combined_prob, ybin_tr)
                # print(f'Loss: {loss.numpy()} with weights: {normalized_weights}')
                return loss1
            
            # Define the loss function for optimizing ensemble weights
            def loss_function(weights1):
                combined_prob = np.zeros_like(prob_combo[0])
                # bias = weights[-num_class:]
                normalized_weights = normal_weight(weights1)
                for j in range(prob_combo[0].shape[1]):
                    for k in range(len(prob_combo)):
                        combined_prob[:, j] += prob_combo[k][:, j] * normalized_weights[j * len(prob_combo) + k]
                # combined_prob += bias
                # combined_prob = np.clip(combined_prob, 1e-15, 1 - 1e-15)  # Avoid log(0) error
                mse_loss = np.mean(np.sum(np.square(ybin_tr - combined_prob), axis = 1))  # Replace cross-entropy with MSE
                return mse_loss

            # Initial guess for weights (even distribution) and biases (zero)
            initial_weights = np.ones(len(prob_combo) * prob_combo[0].shape[1]) / len(prob_combo)
            # initial_biases = np.zeros(num_class)
            # initial_params = np.concatenate([initial_weights, initial_biases])
            
            # Constraints: Weights must sum to 1
            # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w[:len(prob_combo)]) - 1}]
            
            # Bounds: Weights and biases can be any real number
            bounds = [(0, 1) for _ in range(len(initial_weights))]

            
            # print('a: ', id_ind, 'combo: ', ii, ' of ', len(combinations))
            # Optimize the weights and biases
            # result = minimize(loss_function, initial_params, bounds=bounds, constraints=constraints)
            result = minimize(loss_function, initial_weights, method='COBYLA', bounds=bounds)#, bounds=bounds, constraints=constraints)
            
            id_ens[counter, :] = np.array([id_ind] + list(combo), dtype = int)
            fit_ens_tr[counter] = copy.deepcopy(loss_function_other(prob_combo, y_train, result.x))
            weights[counter] = copy.deepcopy(normal_weight(result.x))
            counter += 1
            # If this combination has the best loss, store it
            if loss_function_other(prob_combo, y_train, result.x) < best_loss[id_ind]:
                best_loss[id_ind] = copy.deepcopy(loss_function_other(prob_combo, y_train, result.x))
                best_weights[id_ind] = copy.deepcopy(normal_weight(result.x))
                best_ensemble_prob[id_ind] = np.zeros_like(prob_combo[0])
                
                for j in range(prob_combo[0].shape[1]):
                    for k in range(len(prob_combo)):
                        best_ensemble_prob[id_ind][:, j] += prob_combo[k][:, j] * best_weights[id_ind][j * len(prob_combo) + k]
                        
                # best_ensemble_prob[id_ind] += result.x[-num_class:]
                best_idx[id_ind] = [id_ind] + list(combo)  # Store the indices of the best combination
    
    # Compute all ensemble fitness for validation and test
    if y_val is not None:
        for idx_en in range(id_ens.shape[0]):
            prob_val_combo = [copy.deepcopy(gp.individuals['prob']['isolated']['validation'][p][int(id_ens[idx_en, p])]) for p in range(num_pop)]
            combined_prob_val = np.zeros_like(prob_val_combo[0])
            for j in range(prob_val_combo[0].shape[1]):
                for k in range(len(prob_val_combo)):
                    combined_prob_val[:, j] += weights[id_ind][j * len(prob_val_combo) + k] * prob_val_combo[k][:, j]
            fit_ens_val[id_ind] = loss(combined_prob_val, ybin_val)
    if y_test is not None:
        for idx_en in range(id_ens.shape[0]):
            prob_test_combo = [copy.deepcopy(gp.individuals['prob']['isolated']['test'][p][int(id_ens[idx_en, p])]) for p in range(num_pop)]
            combined_prob_ts = np.zeros_like(prob_test_combo[0])
            for j in range(prob_test_combo[0].shape[1]):
                for k in range(len(prob_test_combo)):
                    combined_prob_ts[:, j] += weights[id_ind][j * len(prob_test_combo) + k] * prob_test_combo[k][:, j]
            fit_ens_ts[id_ind] = loss(combined_prob_ts, ybin_ts)
    
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
        
        for pop_id in range(num_pop):
            num_nodes_en[id_ind].append(copy.deepcopy(num_nodes[pop_id][int(en_idx[id_ind,pop_id])]))
            depth_en[id_ind].append(copy.deepcopy(depth[pop_id][int(en_idx[id_ind,pop_id])]))

        # Repeat for validation and test sets if available
        if y_val is not None:
            prob_val_combo = [copy.deepcopy(gp.individuals['prob']['isolated']['validation'][id_pop][int(best_idx[id_ind][0])])]
            for j, pop_id in enumerate(other_populations):
                prob_val_combo.append(copy.deepcopy(gp.individuals['prob']['isolated']['validation'][pop_id][int(best_idx[id_ind][j+1])]))
            combined_prob_val = np.zeros_like(prob_val_combo[0])
            for k in range(len(prob_val_combo)):
                combined_prob_val += best_weights[id_ind][k] * prob_val_combo[k]
            
            prob_en_val[id_ind] =   copy.deepcopy(combined_prob_val)
            fit_en_val[id_ind] =    loss(combined_prob_val, ybin_val)
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
            fit_en_ts[id_ind] =     loss(combined_prob_ts, ybin_ts)
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
        id_ens,
        fit_ens_tr,
        fit_ens_val,
        fit_ens_ts,
    ]
    return results_en
            
            
            