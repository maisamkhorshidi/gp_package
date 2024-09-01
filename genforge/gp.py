import numpy as np


# Default parameters
DEFAULT_PARAMETERS = {
    'runcontrol_num_pop': None,                         # Number of populations (determined by idx from user input)
    'runcontrol_agg_method': 'Ensemble',                # The Aggreagation Method
    'runcontrol_pop_size': 25,                          # Population size
    'runcontrol_generations': 150,                      # Number of generations
    'runcontrol_stallgen': 20,                          # Terminate if fitness doesn't change after stallgen generations
    'runcontrol_verbose': 1,                            # The generation frequency with which results are printed to Console
    'runcontrol_savefreq': 0,                           # The generation frequency to save the results
    'runcontrol_quiet': False,                          # If true, then GP runs with no console output
    'runcontrol_useparallel': False,                    # true to manually enable parallel CPU fitness evals (requires multiprocessing)
    'runcontrol_n_jobs': 1,                             # if parallel fitness evals enabled, this is the number of "CPUs" to use
    'runcontrol_showBestInputs': True,                  # if true then shows inputs in 'best' individual during run
    'runcontrol_showValBestInputs': False,              # if true then shows inputs in 'valbest' individual during run
    'runcontrol_timeout': np.Inf,                       # gp run will terminate if the run exceeds this values (seconds)
    'runcontrol_runs': 1,                               # number of independent runs to perform and then merge
    'runcontrol_supressConfig': True,                   # true to only evaluate the config file for the first run in a merged multirun
    'runcontrol_usecache': True,                        # fitness caching: used when copying individuals in a gen
    'runcontrol_minimisation': True,                    # True if the problem is minimization and False if it is maximization
    'runcontrol_tolfit': 1e-9,                          # Tolfit means if fitness doesn't change as much as tolfit it is considered not improving
    'runcontrol_plotfitness': True,                     # plot the stats
    'runcontrol_plotrankall': True,
    'runcontrol_plotrankbest': True,
    'runcontrol_plotformat': ['png'],                   # plot format
    'runcontrol_plotfolder': '',                        # plot folder
    'softmax_learning_rate': [0.01],                    # num_pop: The learning rate for the softmax function
    'softmax_optimizer_type': ['adam'],                 # num_pop: the optimizer type: adam, sgd, rmsprop
    'softmax_initializer': ['glorot_uniform'],          # num_pop: the initializer: glorot_uniform, he_normal, random_normal
    'softmax_regularization': [None],                   # num_pop: l1 or l2 regularization
    'softmax_regularization_rate': [0.01],              # num_pop: regularization rate
    'softmax_batch_size': [1],                          # num_pop: batch size
    'softmax_epochs': [30],                             # num_pop: number of epochs
    'softmax_momentum': [0.9],                          # num_pop: momentum, required for sgd optimizer
    'softmax_decay': [1e-6],                            # num_pop: decay rate, required for rmsprop and sgd optimizers
    'softmax_clipnorm': [None],                         # num_pop: clipnorm, optional for any optimizer
    'softmax_clipvalue': [None],                        # num_pop: clipvalue, optional for any optimizer
    'softmax_patience': [10],                           # num_pop: patience epochs for earlystopping
    'selection_tournament_size': [2],                   # num_pop: Tournament size
    'selection_elite_fraction': [0.05],                 # num_pop: Elite fraction in isolated population
    'selection_elite_fraction_ensemble': [0.05],        # num_pop: Elite fraction for all populations
    'selection_tournament_lex_pressure': [True],        # num_pop: set to true to use Sean Luke's et al.'s lexographic selection pressure during regular tournament selection
    'selection_tournament_p_pareto': [0],               # num_pop: probability that a pareto tournament will be used for any given selection event.
    'selection_p_ensemble': [0],                        # num_pop: probability of using ensemble fitness for selection
    'const_p_ERC': [0.1],                               # num_pop: probability of generating an ERC when creating a leaf node
    'const_p_int': [0],                                 # num_pop: probability of generating an integer ERC
    'const_range': [[-10, 10]],                         # num_pop: ERC range 
    'const_num_dec_places': [4],                        # num_pop: decimal places
    'const_about': 'Ephemeral random constants',        # constant nodes generation method
    'functions_name': [['times','minus','plus']],       # num_pop: the functional nodes operators dictionary
    'functions_function': [None],                       # num_pop: the function handles
    'functions_arity': [None],                          # ##arity?
    'functions_active': [None],                         # ##active?
    'operator_p_mutate': [0.14],                        # num_pop: Mutation probability 
    'operator_p_cross': [0.84],                         # num_pop: Crossover probability
    'operator_p_direct': [0.02],                        # num_pop: ##
    'operator_mutate_par': [[0.9,0.05,0.05,0,0,0]],     # num_pop: probability of mutation from [any node, input, constant with guassian distribution]
    'operator_mutate_gaussian_std': [0.1],              # num_pop: for mutate_type 3 (constant perturbation): the standard deviation of the Gaussian used.
    'gene_p_cross_hi': [0.2],                           # num_pop: probability of high level crossover
    'gene_hi_cross_rate': [0.5],                        # num_pop: probability of any given gene being selected during high level crossover
    'gene_multigene': [True],                           # num_pop: Set to true if individuals can have multiple genes
    'gene_max_genes': [5],                              # num_pop: Maximum number of genes per individual
    'tree_build_method': [3],                           # num_pop: 3 = ramped half and half 
    'tree_max_nodes': [np.Inf],                         # num_pop: Maximum nodes that a tree can have
    'tree_max_depth': [4],                              # num_pop: Maximum depth of the trees
    'tree_max_mutate_depth': [4],                       # num_pop: Maximum mutation depth in the trees 
    'fitness_fitfun': 'EnsembleClassification',         # the fitness function to use
    'fitness_terminate': False,                         # true to enable early run termination on attaining a certain fitness value.
    'fitness_terminate_value': -np.Inf,                 # terminate run early if this fitness value or better is achieved
    'fitness_complexityMeasure': 1,                     # 1 = expressional complexity 0 = number of nodes
    'fitness_label': 'Fitness',                         # label for popbrowser etc
    'post_filtered': False,                             # true if population was filtered with GPMODELFILTER
    'post_lastFilter': None,                            # the last GPMODELFILTER to be applied
    'post_merged': 0,                                   # true if this population is the result of merged independent runs
    'post_mergedPopSizes': [],                          # a list of the population sizes that were merged to create the current one
    'post_duplicatesRemoved': False,                    # remove duplicate individuals
    'userdata_name': 'Example GP',                      # Name of the run
    'userdata_stats': True,                             # ## update the stats?
    'userdata_user_fcn': None,                          # ## user function
    'userdata_bootSample': False,                       # boot strap the training data in fitness function
    'userdata_xtrain': None,                            # x train data
    'userdata_ytrain': None,                            # y train data
    'userdata_xval': None,                              # x validation data
    'userdata_yval': None,                              # y validation data
    'userdata_xtest': None,                             # x test data
    'userdata_ytest': None,                             # y test data
    'userdata_numClass': None,                          # number of class labels in ytrain, yval, and ytest 
    'userdata_pop_idx': None,                           # the column index of x in multi-population
    }
    
#     'initial_learning_rate': 0.3,
#     'lambda_L1': 0.001,
#     'lambda_L2': 0.001,
#     'tolgrad': 1e-3,
#     'epsilon': 1e-8,
#     'max_iteration': 500,
#     'ensemble_method': 'ann',
#     'useboruta': False,
#     'borutafraction': 1,
#     'layer': [8, 16, 8],
#     'borutaMaxIter': 10,
#     'borutaNoiseN': 2,
#     'borutalearner': [8],
#     'borutagenes': 5,
#     'interaction': [[1, 0], [0, 1]]
# }

class gpclassifier:
    def __init__(self, **parameters):
        # Merge default parameters with user-provided parameters
        self.parameters = {**DEFAULT_PARAMETERS, **parameters}
        self.runname = self.parameters['userdata_name']
        self.config = {}
        self.userdata = {}
        self.state = {}
        self.fitness = {}
        self.individuals = {}
        self.track = {}
        self.info = {}
        self.cache = {}
        self.population = []
        self.plot = {}
        self.configure()
        self.clearcache()
        # gpcheck(self)  # Check the gp structure

    def configure(self):
        """Configure the GP with specific parameters."""
        if self.parameters['runcontrol_agg_method'].lower() == 'ensemble':
            from .gp_config import gp_config
            from .gp_individuals_ensemble import gp_individuals_ensemble
            from .gp_state_init_ensemble import gp_state_init_ensemble
            from .gp_fitness_init import gp_fitness_init
            gp_config(self)
            gp_individuals_ensemble(self)
            gp_state_init_ensemble(self)
            gp_fitness_init(self)
            
        
    def clearcache(self):
        """ Clear GP Cache """
        if self.config['runcontrol']['usecache']:
            from .gp_cache import gp_cache
            gp_cache(self)
            
    def track_param(self):
        """ Set track parameters"""
        if self.config['runcontrol']['agg_method'].lower() == 'ensemble':
            from .gp_track_param_ensemble import gp_track_param_ensemble
            gp_track_param_ensemble(self)
            
    def build_pop(self):
        from .gp_popbuild_init import gp_popbuild_init
        from .gp_popbuild import gp_popbuild
        if self.state['generation'] == 0:
            """Initialize the population with random genes."""
            gp_popbuild_init(self)
        else:
            """Build new population with the previous one."""
            gp_popbuild(self)

    def evalfitness(self):
        """Evaluate the fitness of individuals stored in the GP structure."""
        if self.config['runcontrol']['parallel']['useparallel']:
            from .gp_evalfitness_par import gp_evalfitness_par
            gp_evalfitness_par(self)
        else:
            from .gp_evalfitness_ord import gp_evalfitness_ord
            gp_evalfitness_ord(self)
            
    def updatestats(self):
        """Updating GP stats"""
        from .gp_updatestats import gp_updatestats
        gp_updatestats(self)
        
    def displaystats(self):
        """Display stats in the console"""
        from .gp_displaystats import gp_displaystats
        gp_displaystats(self)
        
    def plotstats(self):
        """Plot the stats"""
        from .gp_plotfitness import gp_plotfitness
        from .gp_plotrankall import gp_plotrankall
        from .gp_plotrankbest import gp_plotrankbest
        gp_plotfitness(self)
        gp_plotrankall(self)
        gp_plotrankbest(self)
    
    @classmethod
    def initialize(cls, **parameters):
        """Class method to initialize the GP object with parameters."""
        return cls(**parameters)

    def evolve(self):
        """Method to evolve the population."""
        #evolve_process = Evolve(self)
        # The main generation loop
        for gen in range(self.state['generation'] + 1, self.config['runcontrol']['num_generations'] + 1):
            self.state['generation'] = gen
            # Start the generation timer
            # self = gptic(self)
            
            self.track_param()
            self.build_pop()
            self.evalfitness()
            self.updatestats()
            self.clearcache()
            self.displaystats()
            self.plotstats()
            
            if self.state['terminate'] or self.state['run_completed']:
                break
            # evolve_process.evolve()
            # evalfitness(self)  # Evaluate the fitness of the population
            # updatestats(self)  # Update the statistics
            # displaystats(self)  # Display the statistics
            # gptoc(self)  # Update the running time
            # popbuild(self)  # Build the next population
