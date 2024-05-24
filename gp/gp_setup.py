import numpy as np
import os
from .config import configure
from .evolve import Evolve
from .gpinit import gpinit
from .initbuild import initbuild
from .popbuild import popbuild
from .evalfitness import evalfitness
from .mutate import mutate
from .crossover import crossover
from .selection import selection
from .procfuncnodes import procfuncnodes
from .treegen import treegen
from .gptic import gptic
from .gptoc import gptoc
from .gpcheck import gpcheck
from .updatestats import updatestats
from .displaystats import displaystats
from .gpmodelvars import gpmodelvars
from .utils import tree2evalstr, getcomplexity, getnumnodes
from .sigmoid import sigmoid  # Importing the sigmoid function

class GP:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.population = None
        self.config = {}
        self.userdata = {}
        self.state = {}
        self.info = {}
        self.fitness = {'values': None, 'returnvalues': None, 'complexity': None, 'cache': {}}
        self.class_ = {'idx': {}}
        self.config_function = self.configure
        self.configure()
        gpcheck(self)  # Check the gp structure
        gpinit(self)
        procfuncnodes(self)
        self.initialize_population()

    def initialize_population(self):
        """Initialize the population with random genes."""
        pop_size = self.parameters.get('pop_size', 100)
        pop_num = self.config['runcontrol']['num_pop']
        self.population = [[None] * pop_num for _ in range(pop_size)]
        initbuild(self)

    def generate_random_gene(self):
        """Generate a random gene."""
        return treegen(self, 1)  # Example call to treegen

    def configure(self):
        """Configure the GP with specific parameters."""
        global runname, fldres, flddat, xtr1, ytr1, xval1, yval1, xts1, yts1, numclass1
        configure(self, xtr1, ytr1, xval1, yval1, xts1, yts1, runname)

    def binarymapping(self, y, num_classes):
        """Convert y to binary mapping."""
        y_binary = np.zeros((len(y), num_classes))
        for i, val in enumerate(y):
            y_binary[i, val] = 1
        return y_binary

    @staticmethod
    def EnsembleClassification_fitfun(evalstr1, gp):
        """Placeholder for the ensemble classification fitness function."""
        fitness = np.random.rand()
        return fitness, gp

    @classmethod
    def Setup(cls, **parameters):
        """Class method to initialize the GP object with parameters."""
        return cls(**parameters)

    def Evolve(self):
        """Method to evolve the population."""
        evolve_process = Evolve(self)
        for count in range(self.config['runcontrol']['num_gen']):
            self.state['gen'] = count
            self = gptic(self)  # Start the generation timer
            evolve_process.evolve()
            evalfitness(self)  # Evaluate the fitness of the population
            updatestats(self)  # Update the statistics
            displaystats(self)  # Display the statistics
            gptoc(self)  # Update the running time
            popbuild(self)  # Build the next population
