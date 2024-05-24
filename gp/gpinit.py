import numpy as np

def gpinit(gp):
    """Initializes a run."""
    # Determine status of symbolic, parallel and stats toolboxes
    gp.info['toolbox'] = gptoolboxcheck()
    
    # Process function nodes before run
    procfuncnodes(gp)
    
    # Throw an error if there are no inputs, p_ERC=0 and there are no arity zero functions active
    for i in range(len(gp.userdata['xindex'])):
        if (len(gp.userdata['xindex'][i]) == 0 and gp.config['nodes']['const']['p_ERC'][i] == 0 and 
                not any(arity == 0 for arity in gp.config['nodes']['functions']['arity'] if arity in gp.config['nodes']['functions']['active'])):
            raise ValueError('No terminals (inputs, constants or zero arity functions) have been defined for this run.')
    
    # Initialize some state and tracker variables
    gp.state = {
        'count': 1,
        'best': {'fitness': None, 'individual': None},
        'run_completed': False,
        'current_individual': None,
        'std_devfitness': None,
        'terminate': False,
        'force_compute_theta': False,
    }
    gp.fitness['returnvalues'] = [None] * gp.config['runcontrol']['pop_size']

    # Initialize the customized structure for ensemble classification version
    gp.class_ = initialize_class_structure(gp)

    # Process mutation probabilities vector
    gp.operators['mutation']['cumsum_mutate_par'] = np.cumsum(gp.operators['mutation']['mutate_par'])
    
    # Initialize history variables
    gp.results['history'] = {
        'bestfitness': [],
        'meanfitness': [],
        'std_devfitness': [],
        'about': 'Fitness on training data'
    }
    
    # Best of run (on training data) fields
    gp.results['best'] = {
        'fitness': None,
        'individual': None,
        'returnvalues': None,
        'foundatgen': None,
        'about': 'Best individual on training data'
    }

    # Assign field holding fitnesses to gp structure
    gp.fitness['values'] = np.zeros(gp.config['runcontrol']['pop_size'])
    gp.fitness['complexity'] = np.zeros(gp.config['runcontrol']['pop_size'])

    # Cache init
    if gp.config['runcontrol']['usecache']:
        initcache(gp)
    
    return gp

def initialize_class_structure(gp):
    """Initialize the customized structure for ensemble classification."""
    pop_size = gp.config['runcontrol']['pop_size']
    num_pop = gp.config['runcontrol']['num_pop']
    num_gen = gp.config['runcontrol']['num_gen']

    class_ = {
        'pop': [[None] * num_pop for _ in range(pop_size)],
        'net': [[None] * num_pop for _ in range(pop_size)],
        'weight_genes': [[None] * num_pop for _ in range(pop_size)],
        'geneOutput_train': [[None] * num_pop for _ in range(pop_size)],
        'geneOutput_validation': [[None] * num_pop for _ in range(pop_size)],
        'geneOutput_test': [[None] * num_pop for _ in range(pop_size)],
        'gradient_indiv': [[None] * num_pop for _ in range(pop_size)],
        'probindiv_train': [[None] * num_pop for _ in range(pop_size)],
        'probindiv_validation': [[None] * num_pop for _ in range(pop_size)],
        'probindiv_test': [[None] * num_pop for _ in range(pop_size)],
        'lossindiv_train': [[None] * num_pop for _ in range(pop_size)],
        'lossindiv_validation': [[None] * num_pop for _ in range(pop_size)],
        'lossindiv_test': [[None] * num_pop for _ in range(pop_size)],
        'ypindiv_train': [[None] * num_pop for _ in range(pop_size)],
        'ypindiv_valid': [[None] * num_pop for _ in range(pop_size)],
        'ypindiv_test': [[None] * num_pop for _ in range(pop_size)],
        'weight_ensemble': [None] * pop_size,
        'gradient_ensemble': [None] * pop_size,
        'fitness_train_ensemble': np.full(pop_size, np.inf),
        'fitness_validation_ensemble': np.full(pop_size, np.inf),
        'fitness_test_ensemble': np.full(pop_size, np.inf),
        'prob_train_ensemble': [None] * pop_size,
        'prob_validation_ensemble': [None] * pop_size,
        'prob_test_ensemble': [None] * pop_size,
        'loss_train_ensemble': [None] * pop_size,
        'loss_valid_ensemble': [None] * pop_size,
        'loss_test_ensemble': [None] * pop_size,
        'yp_train_ensemble': [None] * pop_size,
        'yp_valid_ensemble': [None] * pop_size,
        'yp_test_ensemble': [None] * pop_size,
        'complexity': np.zeros((pop_size, num_pop)),
        'fitnessindiv': np.zeros((pop_size, num_pop)),
        'idx': [np.tile(np.arange(1, pop_size + 1), (1, num_pop))]
    }

    for i in range(pop_size):
        for j in range(num_pop):
            class_['weight_genes'][i][j] = [None] * num_gen
            class_['gradient_indiv'][i][j] = [None] * num_gen
            class_['lossindiv_train'][i][j] = np.full(num_gen, np.nan)
            class_['lossindiv_validation'][i][j] = np.full(num_gen, np.nan)
            class_['lossindiv_test'][i][j] = np.full(num_gen, np.nan)

    for i in range(pop_size):
        class_['weight_ensemble'][i] = [None] * num_gen
        class_['gradient_ensemble'][i] = [None] * num_gen
        class_['loss_train_ensemble'][i] = np.full(num_gen, np.nan)
        class_['loss_valid_ensemble'][i] = np.full(num_gen, np.nan)
        class_['loss_test_ensemble'][i] = np.full(num_gen, np.nan)

    return class_

def gptoolboxcheck():
    """Mock function to check for toolboxes."""
    return {'symbolic': True, 'parallel': True, 'stats': True}

def procfuncnodes(gp):
    """Process required function node information prior to a run."""
    gp.config['nodes']['functions']['arity'] = []
    for func_name in gp.config['nodes']['functions']['name']:
        arity = func_arity(func_name)
        
        if arity == -1:
            raise ValueError(f"The function {func_name} may not be used as a function node because it has a variable number of arguments.")

        gp.config['nodes']['functions']['arity'].append(arity)
    
    if 'active' not in gp.config['nodes']['functions'] or not gp.config['nodes']['functions']['active']:
        gp.config['nodes']['functions']['active'] = [True] * len(gp.config['nodes']['functions']['name'])

    gp.config['nodes']['functions']['active'] = np.array(gp.config['nodes']['functions']['active'], dtype=bool)

    num_active = np.sum(gp.config['nodes']['functions']['active'])
    if num_active > 22:
        raise ValueError('Maximum number of active functions allowed is 22')

    charnum = 96
    skip = 0
    afid = []
    for i in range(num_active):
        while True:
            if (charnum + i + skip) in [101, 105, 106, 120]:
                skip += 1
            else:
                break
        afid.append(chr(charnum + i + skip))

    gp.config['nodes']['functions']['afid'] = afid

    active_names = [name.upper() for idx, name in enumerate(gp.config['nodes']['functions']['name']) if gp.config['nodes']['functions']['active'][idx]]
    gp.config['nodes']['functions']['active_name_UC'] = active_names

    active_ar = np.array(gp.config['nodes']['functions']['arity'])[gp.config['nodes']['functions']['active']]
    gp.config['nodes']['functions']['afid_argt0'] = [afid[i] for i in range(len(afid)) if active_ar[i] > 0]
    gp.config['nodes']['functions']['afid_areq0'] = [afid[i] for i in range(len(afid)) if active_ar[i] == 0]
    gp.config['nodes']['functions']['arity_argt0'] = active_ar[active_ar > 0]

    gp.config['nodes']['functions']['fun_lengthargt0'] = len(gp.config['nodes']['functions']['afid_argt0'])
    gp.config['nodes']['functions']['fun_lengthareq0'] = len(gp.config['nodes']['functions']['afid_areq0'])

def initcache(gp):
    """Sets up fitness cache."""
    gp.fitness['cache'] = {}

def func_arity(func_name):
    """Determine the arity of a function."""
    # Placeholder: In a real implementation, this would check the actual function arity.
    # For now, we return a mock value of 2.
    return 2
