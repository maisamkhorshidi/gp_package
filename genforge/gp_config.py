import numpy as np
import os
import importlib
import inspect

def import_functions(function_name):
    module_name = function_name.lower()  # Assuming module name matches function name
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    
    # Get the function's signature
    signature = inspect.signature(function)
    
    # Count the number of required arguments
    num_args = sum(
        1 for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    
    return function, num_args

def binarymapping(y, num_classes):
    """Convert y to binary mapping."""
    y_binary = np.zeros((len(y), num_classes))
    for i, val in enumerate(y):
        y_binary[i, val] = 1
    return y_binary
##################################################################
def gp_config(gp):
    ## Checking for errors in the parameters
    # Training X
    if isinstance(gp.parameters['userdata_xtrain'],np.ndarray):
        xtr1 = gp.parameters['userdata_xtrain']
    else:
        raise ValueError("Training input data should be numpy array. Set userdata_xtrain ...")
        
    # Training Y
    if isinstance(gp.parameters['userdata_ytrain'],np.ndarray):
        ytr = gp.parameters['userdata_ytrain']
    else:
        raise ValueError("Training output data should be numpy array. Set userdata_ytrain ...")
        
    # Taining data size
    if len(xtr1) != len(ytr):
        raise ValueError("The length of training input and output data should be the same. Set userdata_xtrain and userdata_ytrain ...")
        
    # Validation data
    if gp.parameters['userdata_xval'] is None:
        xval1 = None
        yval = None
    elif not isinstance(gp.parameters['userdata_xval'],np.ndarray):
        raise ValueError("Validation input data should be numpy array. Set userdata_xval ...")
    elif gp.parameters['userdata_xval'].shape[1] != gp.parameters['userdata_xtrain'].shape[1]:
        raise ValueError("Number of columns in validation input data should be as same as training data. Set userdata_xval ...")
    elif len(gp.parameters['userdata_xval']) != len(gp.parameters['userdata_yval']):
        raise ValueError("The length of validation input and output data should be the same. Set userdata_xval and userdata_yval ...")
    else:
        xval1 = gp.parameters['userdata_xval']
        yval = gp.parameters['userdata_yval']
        
    # Test data
    if gp.parameters['userdata_xtest'] is None:
        xts1 = None
        yts = None
    elif not isinstance(gp.parameters['userdata_xtest'],np.ndarray):
        raise ValueError("Testing input data should be numpy array. Set userdata_xtest ...")
    elif gp.parameters['userdata_xtest'].shape[1] != gp.parameters['userdata_xtrain'].shape[1]:
        raise ValueError("Number of columns in testing input data should be as same as training data. Set userdata_xtest ...")
    elif len(gp.parameters['userdata_xtest']) != len(gp.parameters['userdata_ytest']):
        raise ValueError("The length of testing input and output data should be the same. Set userdata_xtest and userdata_ytest ...")
    else:
        xts1 = gp.parameters['userdata_xtest']
        yts = gp.parameters['userdata_ytest']
    
    ## Setting the parameters
    # Population size
    if gp.parameters['userdata_pop_idx'] is None and gp.parameters['runcontrol_num_pop'] == 1:
        num_pop = 1
        xtr = xtr1
        xval = xval1
        xts = xts1
        pop_idx = [list(range(xtr1.shape[1]))]
    elif gp.parameters['userdata_pop_idx'] is not None and gp.parameters['runcontrol_num_pop'] == 1:
        num_pop = 1
        xtr = xtr1
        xval = xval1
        xts = xts1
        pop_idx = gp.parameters['userdata_pop_idx']
    elif gp.parameters['userdata_pop_idx'] is None and gp.parameters['runcontrol_num_pop'] > 1:
        raise ValueError("The userdata_pop_idx could not be empty for multipopulation GP. Set userdata_pop_idx ...")
    elif gp.parameters['userdata_pop_idx'] is not None:
        num_pop = len(gp.parameters['userdata_pop_idx'])
        xtr = xtr1
        xval = xval1
        xts = xts1
        pop_idx = list()
        for val in gp.parameters['userdata_pop_idx']:
            pop_idx.append(val)
            
    
    ## Softmax parameters
    gp.config['softmax'] = {}
    
    if len(gp.parameters['softmax_learning_rate']) == 1:
        gp.config['softmax']['learning_rate'] = gp.parameters['softmax_learning_rate'] * num_pop
    elif len(gp.parameters['softmax_learning_rate']) != num_pop:
        raise ValueError("The length of softmax_learning_rate should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['learning_rate'] = gp.parameters['softmax_learning_rate']
    
    if len(gp.parameters['softmax_optimizer_type']) == 1:
        gp.config['softmax']['optimizer_type'] = gp.parameters['softmax_optimizer_type'] * num_pop
    elif len(gp.parameters['softmax_optimizer_type']) != num_pop:
        raise ValueError("The length of softmax_optimizer_type should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['optimizer_type'] = gp.parameters['softmax_optimizer_type']
        
    if len(gp.parameters['softmax_initializer']) == 1:
        gp.config['softmax']['initializer'] = gp.parameters['softmax_initializer'] * num_pop
    elif len(gp.parameters['softmax_initializer']) != num_pop:
        raise ValueError("The length of softmax_initializer should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['initializer'] = gp.parameters['softmax_initializer']
        
    if len(gp.parameters['softmax_regularization']) == 1:
        gp.config['softmax']['regularization'] = gp.parameters['softmax_regularization'] * num_pop
    elif len(gp.parameters['softmax_regularization']) != num_pop:
        raise ValueError("The length of softmax_regularization should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['regularization'] = gp.parameters['softmax_regularization']
        
    if len(gp.parameters['softmax_regularization_rate']) == 1:
        gp.config['softmax']['regularization_rate'] = gp.parameters['softmax_regularization_rate'] * num_pop
    elif len(gp.parameters['softmax_regularization_rate']) != num_pop:
        raise ValueError("The length of softmax_regularization_rate should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['regularization_rate'] = gp.parameters['softmax_regularization_rate']
        
    if len(gp.parameters['softmax_batch_size']) == 1:
        gp.config['softmax']['batch_size'] = gp.parameters['softmax_batch_size'] * num_pop
    elif len(gp.parameters['softmax_batch_size']) != num_pop:
        raise ValueError("The length of softmax_batch_size should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['batch_size'] = gp.parameters['softmax_batch_size']
        
    if len(gp.parameters['softmax_epochs']) == 1:
        gp.config['softmax']['epochs'] = gp.parameters['softmax_epochs'] * num_pop
    elif len(gp.parameters['softmax_epochs']) != num_pop:
        raise ValueError("The length of softmax_epochs should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['epochs'] = gp.parameters['softmax_epochs']
        
    if len(gp.parameters['softmax_momentum']) == 1:
        gp.config['softmax']['momentum'] = gp.parameters['softmax_momentum'] * num_pop
    elif len(gp.parameters['softmax_momentum']) != num_pop:
        raise ValueError("The length of softmax_momentum should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['momentum'] = gp.parameters['softmax_momentum']
        
    if len(gp.parameters['softmax_decay']) == 1:
        gp.config['softmax']['decay'] = gp.parameters['softmax_decay'] * num_pop
    elif len(gp.parameters['softmax_decay']) != num_pop:
        raise ValueError("The length of softmax_decay should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['decay'] = gp.parameters['softmax_decay']
    
    if len(gp.parameters['softmax_clipnorm']) == 1:
        gp.config['softmax']['clipnorm'] = gp.parameters['softmax_clipnorm'] * num_pop
    elif len(gp.parameters['softmax_clipnorm']) != num_pop:
        raise ValueError("The length of softmax_clipnorm should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['clipnorm'] = gp.parameters['softmax_clipnorm']
    
    if len(gp.parameters['softmax_clipvalue']) == 1:
        gp.config['softmax']['clipvalue'] = gp.parameters['softmax_clipvalue'] * num_pop
    elif len(gp.parameters['softmax_clipvalue']) != num_pop:
        raise ValueError("The length of softmax_clipvalue should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['clipvalue'] = gp.parameters['softmax_clipvalue']
    
    if len(gp.parameters['softmax_patience']) == 1:
        gp.config['softmax']['patience'] = gp.parameters['softmax_patience'] * num_pop
    elif len(gp.parameters['softmax_patience']) != num_pop:
        raise ValueError("The length of softmax_patience should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['softmax']['patience'] = gp.parameters['softmax_patience']
    
    
    ## Runcontrol parameters
    gp.config['runcontrol'] = {
        'num_pop': num_pop,
        'pop_size': gp.parameters['runcontrol_pop_size'],
        'num_class': len(np.unique(np.concatenate((ytr, yval, yts)))),
        'num_generations': gp.parameters['runcontrol_generations'],
        'stallgen': gp.parameters['runcontrol_stallgen'],
        'tolfit': gp.parameters['runcontrol_tolfit'],
        'verbose': gp.parameters['runcontrol_verbose'],
        'savefreq': gp.parameters['runcontrol_savefreq'],
        'quiet': gp.parameters['runcontrol_quiet'],
        'parallel': None,
        'showBestInputs': gp.parameters['runcontrol_showBestInputs'],
        'showValBestInputs': gp.parameters['runcontrol_showValBestInputs'],
        'timeout': gp.parameters['runcontrol_timeout'],
        'runs': gp.parameters['runcontrol_runs'],
        'supressConfig': gp.parameters['runcontrol_supressConfig'],
        'usecache': gp.parameters['runcontrol_usecache'],
        'minimisation': gp.parameters['runcontrol_minimisation'],
        'plot': {
            'fitness': gp.parameters['runcontrol_plotfitness'],
            'rankall': gp.parameters['runcontrol_plotrankall'],
            'rankbest': gp.parameters['runcontrol_plotrankbest'],
            'format': gp.parameters['runcontrol_plotformat'],
            'folder': gp.parameters['runcontrol_plotfolder'],
            },
        }
    
    if num_pop > 1 and gp.parameters['runcontrol_agg_method'] is None:
        gp.config['runcontrol']['agg_method'] = 'Ensemble'
    else:
        gp.config['runcontrol']['agg_method'] = gp.parameters['runcontrol_agg_method']
    
    ## Parallel parameters
    gp.config['runcontrol']['parallel'] = {
        'useparallel': gp.parameters['runcontrol_useparallel'],
        'n_jobs': gp.parameters['runcontrol_n_jobs'],
        }
    
    ## Selection Parameters
    gp.config['selection'] = {}
    
    if len(gp.parameters['selection_tournament_size']) == 1:
        gp.config['selection']['tournament_size'] = gp.parameters['selection_tournament_size'] * num_pop
    elif len(gp.parameters['selection_tournament_size']) != num_pop:
        raise ValueError("The length of selection_tournament_size should be as same as the runcontrol_num_pop. Set selection_tournament_size ...")
    else:
        gp.config['selection']['tournament_size'] = gp.parameters['selection_tournament_size']
    
    if len(gp.parameters['selection_elite_fraction']) == 1:
        gp.config['selection']['elite_fraction'] = gp.parameters['selection_elite_fraction'] * num_pop
    elif len(gp.parameters['selection_elite_fraction']) != num_pop:
        raise ValueError("The length of selection_elite_fraction should be as same as the runcontrol_num_pop. Set selection_elite_fraction ...")
    else:
        gp.config['selection']['elite_fraction'] = gp.parameters['selection_elite_fraction']

    gp.config['selection']['elite_fraction_ensemble'] = gp.parameters['selection_elite_fraction_ensemble']
    
    if len(gp.parameters['selection_tournament_lex_pressure']) == 1:
        gp.config['selection']['tournament_lex_pressure'] = gp.parameters['selection_tournament_lex_pressure'] * num_pop
    elif len(gp.parameters['selection_tournament_lex_pressure']) != num_pop:
        raise ValueError("The length of selection_tournament_lex_pressure should be as same as the runcontrol_num_pop. Set selection_tournament_lex_pressure ...")
    else:
        gp.config['selection']['tournament_lex_pressure'] = gp.parameters['selection_tournament_lex_pressure']
    
    if len(gp.parameters['selection_tournament_p_pareto']) == 1:
        gp.config['selection']['tournament_p_pareto'] = gp.parameters['selection_tournament_p_pareto'] * num_pop
    elif len(gp.parameters['selection_tournament_p_pareto']) != num_pop:
        raise ValueError("The length of selection_tournament_p_pareto should be as same as the runcontrol_num_pop. Set selection_tournament_p_pareto ...")
    else:
        gp.config['selection']['tournament_p_pareto'] = gp.parameters['selection_tournament_p_pareto']
        
    if len(gp.parameters['selection_p_ensemble']) == 1:
        gp.config['selection']['p_ensemble'] = gp.parameters['selection_p_ensemble'] * num_pop
    elif len(gp.parameters['selection_p_ensemble']) != num_pop:
        raise ValueError("The length of selection_p_ensemble should be as same as the runcontrol_num_pop. Set selection_p_ensemble ...")
    else:
        gp.config['selection']['p_ensemble'] = gp.parameters['selection_p_ensemble']
    
    ## Node Parameters
    ## Constants
    gp.config['nodes'] = {}
    gp.config['nodes']['const'] = {}
    gp.config['nodes']['const']['about'] = gp.parameters['const_about']
    
    if len(gp.parameters['const_p_ERC']) == 1:
        gp.config['nodes']['const']['p_ERC'] = gp.parameters['const_p_ERC'] * num_pop
    elif len(gp.parameters['const_p_ERC']) != num_pop:
        raise ValueError("The length of const_p_ERC should be as same as the runcontrol_num_pop. Set const_p_ERC ...")
    else:
        gp.config['nodes']['const']['p_ERC'] = gp.parameters['const_p_ERC']
    
    if len(gp.parameters['const_p_int']) == 1:
        gp.config['nodes']['const']['p_int'] = gp.parameters['const_p_int'] * num_pop
    elif len(gp.parameters['const_p_int']) != num_pop:
        raise ValueError("The length of const_p_int should be as same as the runcontrol_num_pop. Set const_p_int ...")
    else:
        gp.config['nodes']['const']['p_int'] = gp.parameters['const_p_int']
        
    if len(gp.parameters['const_range']) == 1:
        gp.config['nodes']['const']['range'] = gp.parameters['const_range'] * num_pop
    elif len(gp.parameters['const_range']) != num_pop:
        raise ValueError("The length of const_range should be as same as the runcontrol_num_pop. Set const_range ...")
    else:
        gp.config['nodes']['const']['range'] = gp.parameters['const_range']
    
    if len(gp.parameters['const_num_dec_places']) == 1:
        gp.config['nodes']['const']['num_dec_places'] = gp.parameters['const_num_dec_places'] * num_pop
    elif len(gp.parameters['const_num_dec_places']) != num_pop:
        raise ValueError("The length of const_num_dec_places should be as same as the runcontrol_num_pop. Set const_num_dec_places ...")
    else:
        gp.config['nodes']['const']['num_dec_places'] = gp.parameters['const_num_dec_places']
    
    ## Functions
    gp.config['nodes']['functions'] = {}
    
    if len(gp.parameters['functions_name']) == 1:
        gp.config['nodes']['functions']['name'] = gp.parameters['functions_name'] * num_pop
    elif len(gp.parameters['functions_name']) != num_pop:
        raise ValueError("The length of functions_name should be as same as the runcontrol_num_pop. Set functions_name ...")
    else:
        gp.config['nodes']['functions']['name'] = gp.parameters['functions_name']
        
    gp.config['nodes']['functions']['function'] = []
    gp.config['nodes']['functions']['arity'] = []
    for item in gp.config['nodes']['functions']['name']:
        function_dict = {}
        arity_list = []
        for val in item:
            function, num_args = import_functions(val)
            function_dict[val] = function
            arity_list.append(num_args)
        gp.config['nodes']['functions']['function'].append(function_dict)
        gp.config['nodes']['functions']['arity'].append(arity_list)

    gp.config['nodes']['functions']['active'] = [[1 for val in item] for item in gp.config['nodes']['functions']['name']]
    
    ## Operator parameter
    gp.config['operator'] = {}
    
    if len(gp.parameters['operator_p_mutate']) == 1:
        gp.config['operator']['p_mutate'] = gp.parameters['operator_p_mutate'] * num_pop
    elif len(gp.parameters['operator_p_mutate']) != num_pop:
        raise ValueError("The length of operator_p_mutate should be as same as the runcontrol_num_pop. Set operator_p_mutate ...")
    else:
        gp.config['operator']['p_mutate'] = gp.parameters['operator_p_mutate']
    
    if len(gp.parameters['operator_p_cross']) == 1:
        gp.config['operator']['p_cross'] = gp.parameters['operator_p_cross'] * num_pop
    elif len(gp.parameters['operator_p_cross']) != num_pop:
        raise ValueError("The length of operator_p_cross should be as same as the runcontrol_num_pop. Set operator_p_cross ...")
    else:
        gp.config['operator']['p_cross'] = gp.parameters['operator_p_cross']
    
    if len(gp.parameters['operator_p_direct']) == 1:
        gp.config['operator']['p_direct'] = gp.parameters['operator_p_direct'] * num_pop
    elif len(gp.parameters['operator_p_direct']) != num_pop:
        raise ValueError("The length of operator_p_direct should be as same as the runcontrol_num_pop. Set operator_p_direct ...")
    else:
        gp.config['operator']['p_direct'] = gp.parameters['operator_p_direct']
    
    if len(gp.parameters['operator_mutate_par']) == 1:
        gp.config['operator']['mutate_par'] = gp.parameters['operator_mutate_par'] * num_pop
        # Process mutation probabilities vector
        gp.config['operator']['mutate_par_cumsum'] = [np.cumsum(gp.parameters['operator_mutate_par'])] * num_pop
    elif len(gp.parameters['operator_mutate_par']) != num_pop:
        raise ValueError("The length of operator_p_direct should be as same as the runcontrol_num_pop. Set operator_mutate_par ...")
    else:
        gp.config['operator']['mutate_par'] = gp.parameters['operator_mutate_par']
        gp.config['operator']['mutate_par_cumsum'] = [np.cumsum(val) for val in gp.parameters['operator_mutate_par']]
    
    if len(gp.parameters['operator_mutate_gaussian_std']) == 1:
        gp.config['operator']['mutate_gaussian_std'] = gp.parameters['operator_mutate_gaussian_std'] * num_pop
    elif len(gp.parameters['operator_mutate_gaussian_std']) != num_pop:
        raise ValueError("The length of operator_mutate_gaussian_std should be as same as the runcontrol_num_pop. Set operator_mutate_gaussian_std ...")
    else:
        gp.config['operator']['mutate_gaussian_std'] = gp.parameters['operator_mutate_gaussian_std']
    
    ## Gene parameters
    gp.config['gene'] = {}
    
    if len(gp.parameters['gene_p_cross_hi']) == 1:
        gp.config['gene']['p_cross_hi'] = gp.parameters['gene_p_cross_hi'] * num_pop
    elif len(gp.parameters['gene_p_cross_hi']) != num_pop:
        raise ValueError("The length of gene_p_cross_hi should be as same as the runcontrol_num_pop. Set gene_p_cross_hi ...")
    else:
        gp.config['gene']['gene_p_cross_hi'] = gp.parameters['gene_p_cross_hi']
    
    if len(gp.parameters['gene_hi_cross_rate']) == 1:
        gp.config['gene']['hi_cross_rate'] = gp.parameters['gene_hi_cross_rate'] * num_pop
    elif len(gp.parameters['gene_hi_cross_rate']) != num_pop:
        raise ValueError("The length of gene_hi_cross_rate should be as same as the runcontrol_num_pop. Set gene_hi_cross_rate ...")
    else:
        gp.config['gene']['hi_cross_rate'] = gp.parameters['gene_hi_cross_rate']
    
    if len(gp.parameters['gene_multigene']) == 1:
        gp.config['gene']['multigene'] = gp.parameters['gene_multigene'] * num_pop
    elif len(gp.parameters['gene_multigene']) != num_pop:
        raise ValueError("The length of gene_multigene should be as same as the runcontrol_num_pop. Set gene_multigene ...")
    else:
        gp.config['gene']['multigene'] = gp.parameters['gene_multigene']
    
    if len(gp.parameters['gene_max_genes']) == 1:
        gp.config['gene']['max_genes'] = gp.parameters['gene_max_genes'] * num_pop
    elif len(gp.parameters['gene_max_genes']) != num_pop:
        raise ValueError("The length of gene_max_genes should be as same as the runcontrol_num_pop. Set gene_max_genes ...")
    else:
        gp.config['gene']['max_genes'] = gp.parameters['gene_max_genes']
    
    ## Tree parameters
    gp.config['tree']={}
    
    if len(gp.parameters['tree_build_method']) == 1:
        gp.config['tree']['build_method'] = gp.parameters['tree_build_method'] * num_pop
    elif len(gp.parameters['tree_build_method']) != num_pop:
        raise ValueError("The length of tree_build_method should be as same as the runcontrol_num_pop. Set tree_build_method ...")
    else:
        gp.config['tree']['build_method'] = gp.parameters['tree_build_method']
        
    if len(gp.parameters['tree_max_nodes']) == 1:
        gp.config['tree']['max_nodes'] = gp.parameters['tree_max_nodes'] * num_pop
    elif len(gp.parameters['tree_max_nodes']) != num_pop:
        raise ValueError("The length of tree_max_nodes should be as same as the runcontrol_num_pop. Set tree_max_nodes ...")
    else:
        gp.config['tree']['max_nodes'] = gp.parameters['tree_max_nodes']
    
    if len(gp.parameters['tree_max_depth']) == 1:
        gp.config['tree']['max_depth'] = gp.parameters['tree_max_depth'] * num_pop
    elif len(gp.parameters['tree_max_depth']) != num_pop:
        raise ValueError("The length of tree_build_method should be as same as the runcontrol_num_pop. Set tree_max_depth ...")
    else:
        gp.config['tree']['max_depth'] = gp.parameters['tree_max_depth']
    
    if len(gp.parameters['tree_max_mutate_depth']) == 1:
        gp.config['tree']['max_mutate_depth'] = gp.parameters['tree_max_mutate_depth'] * num_pop
    elif len(gp.parameters['tree_max_mutate_depth']) != num_pop:
        raise ValueError("The length of tree_max_mutate_depth should be as same as the runcontrol_num_pop. Set tree_max_mutate_depth ...")
    else:
        gp.config['tree']['max_mutate_depth'] = gp.parameters['tree_max_mutate_depth']
    
    ## Fitness parameters
    gp.config['fitness'] = {
        'fitfun': 1,#import_functions(gp.parameters['fitness_fitfun']),
        'terminate': gp.parameters['fitness_terminate'],
        'terminate_value': gp.parameters['fitness_terminate_value'],
        'complexityMeasure': gp.parameters['fitness_complexityMeasure'],
        'Label': gp.parameters['fitness_label']
        }
    
    ## Post processing parameters
    gp.config['post'] = {
        'filtered': gp.parameters['post_filtered'],
        'lastFilter': gp.parameters['post_lastFilter'],
        'merged': gp.parameters['post_merged'],
        'mergedPopSizes': gp.parameters['post_mergedPopSizes'],
        'duplicatesRemoved': gp.parameters['post_duplicatesRemoved']
        }

    ## Userdata parameters
    gp.userdata['name'] = gp.parameters['userdata_name']
    gp.userdata['stats'] = gp.parameters['userdata_stats']
    gp.userdata['user_fcn'] = gp.parameters['userdata_user_fcn']
    gp.userdata['bootSample'] = gp.parameters['userdata_bootSample']
    gp.userdata['pop_idx'] = pop_idx.copy()
    gp.userdata['num_class'] = len(np.unique(np.concatenate((ytr, yval, yts))))
    gp.userdata['xtrain'] = xtr.copy()
    gp.userdata['ytrain'] = ytr.copy()
    gp.userdata['xval'] = xval.copy()
    gp.userdata['yval'] = yval.copy()
    gp.userdata['xtest'] = xts.copy()
    gp.userdata['ytest'] = yts.copy()
    gp.userdata['ybinarytrain'] = binarymapping(ytr, gp.userdata['num_class'])
    gp.userdata['ybinaryval'] = binarymapping(yval, gp.userdata['num_class'])
    gp.userdata['ybinarytest'] = binarymapping(yts, gp.userdata['num_class'])
    gp.userdata['PlotName'] = gp.userdata['name'] + '.png'
    
    

