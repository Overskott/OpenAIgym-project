from src.genotypes import CellularAutomaton1D, NeuralNetwork
import src.config as config
import numpy as np


class Generation(object):
    """ A class for managing and organizing phenotypes in a generation.

        Attributes:
            genotype (str): The type of genotype to populate the generation.
            generation_number (str): The generation identificatior
            generation_size (int): Number of phenotypes in the generation.
    """

    def __init__(self, genotype, id_number, population=None):
        """ Initializing a Generation instance.

            Args:
                genotype (str): Type of phenotype in the generation. 'ca' for CA
                or 'nn' for NN is accepted.
                generation_number (str): An identificator for the generation.
                generation_size (int): Number of individuals in the generation
        """
        self.genotype = genotype
        self.generation_number = id_number
        self.generation_size = config.data['evolution']['generation_size']

        if population is None:
            self.initialize_population()
        else:
            self.population = np.copy(population)

    def __getitem__(self, i):
        return self.population[i]

    def __len__(self):
        return len(self.population)

    @property
    def genotype(self):
        return self._genotype

    @genotype.setter
    def genotype(self, genotype):
        self._genotype = genotype

    def initialize_population(self):
        """ Generates an initial population"""
        population = []

        if self.genotype == 'ca':
            for i in range(self.generation_size):
                phenotype = CellularAutomaton1D(f"{self.generation_number}-{i+1}")
                population.append(phenotype)

        elif self.genotype == 'nn':
            for i in range(self.generation_size):
                phenotype = NeuralNetwork(f"{self.generation_number}-{i+1}")
                population.append(phenotype)
                pass

        self.population = population

    def get_population_fitness(self):
        """ Return a list of each phenotype's fitness"""
        return np.asarray([phenotype.get_fitness() for phenotype in self.population])

    def sort_population_by_fitness(self):
        """ Order the individuals in population from lowest to highest fitness"""
        fitness_dict = {f: f.get_fitness() for f in self.population}
        sorted_dict = sorted(fitness_dict.items(), key=lambda x: x[1])
        self.population = [c[0] for c in sorted_dict]
