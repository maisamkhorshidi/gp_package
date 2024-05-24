import numpy as np

def selection(gp, jj=1):
    """Select an individual from the current population."""
    if np.random.rand() >= gp.config['selection']['tournament']['p_pareto']:
        # Method A: Standard fitness-based tournament
        tournament_size = gp.config['selection']['tournament']['size'][jj]
        competitors = np.random.choice(gp.config['runcontrol']['pop_size'], tournament_size, replace=True)
        
        fitness_values = gp.class_['fitnessindiv'][competitors, jj]
        
        if gp.config['fitness']['minimisation']:
            best_fitness = np.min(fitness_values)
        else:
            best_fitness = np.max(fitness_values)
        
        best_indices = np.where(fitness_values == best_fitness)[0]
        
        if len(best_indices) > 1 and gp.config['selection']['tournament']['lex_pressure']:
            complexities = gp.class_['complexity'][competitors[best_indices], jj]
            best_indices = best_indices[np.argmin(complexities)]
        else:
            best_indices = np.random.choice(best_indices)
        
        ID = competitors[best_indices]
        
    else:
        # Method B: Pareto tournament
        tournament_size = gp.config['selection']['tournament']['size'][jj]
        competitors = np.random.choice(gp.config['runcontrol']['pop_size'], tournament_size, replace=True)
        
        fitness_values = gp.class_['fitnessindiv'][competitors, jj]
        complexities = gp.class_['complexity'][competitors, jj]
        
        if gp.config['fitness']['minimisation']:
            mo = np.column_stack((fitness_values, complexities))
        else:
            mo = np.column_stack((-1 * fitness_values, complexities))
        
        mo = mo[~np.isinf(mo[:, 0])]
        competitors = competitors[~np.isinf(mo[:, 0])]
        
        if len(mo) == 0:
            return selection(gp, jj)
        
        rank = ndfsort_rank1(mo)
        rank1_indices = np.where(rank == 1)[0]
        ID = np.random.choice(competitors[rank1_indices])
    
    return ID

def ndfsort_rank1(mo):
    """Perform non-dominated sorting to find rank 1 individuals."""
    is_dominated = np.zeros(len(mo), dtype=bool)
    for i in range(len(mo)):
        for j in range(len(mo)):
            if all(mo[j] <= mo[i]) and any(mo[j] < mo[i]):
                is_dominated[i] = True
                break
    rank1 = np.where(~is_dominated)[0]
    return rank1
