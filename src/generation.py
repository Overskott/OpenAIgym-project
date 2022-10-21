import copy
from typing import List

import numpy as np

from genotypes import Genotype, CellularAutomaton1D
import config
from utils.utils import *
import evolution


class Generation(object):

    generation_size: int

    def __init__(self, id_number, population=None):

        if population is None:
            self.initialize_population()
        else:
            self.population = copy.deepcopy(population)

        self.id_number = id_number
        self.generation_size = config.data['generation_size']

    def __getitem__(self, i):
        return self.population[i]

    # def random_rule(self):
    #     return np.random.randint(0, 2, 2 ** config.data['ca_hood_size'], dtype='u4')

    def initialize_population(self):
        population = []

        for i in range(config.data['generation_size']):
            rule = evolution.random_bitarray(2 ** config.data['ca_hood_size'])
            size = np.random.randint(50, 150)
            steps = np.random.randint(10, 75)
            phenotype = CellularAutomaton1D(rule, size, steps)
            population.append(phenotype)

        self.population = population

    def get_population_fitness(self):
        return np.asarray([phenotype.get_fitness() for phenotype in self.population])

    # def select_parents(self) -> List[Genotype]:
    #     self.sort_population_by_fitness()
    #     return self.population[-(self.generation_size >> 1):]
    #
    def sort_population_by_fitness(self):
        fitness_dict = {f: f.get_fitness() for f in self.population}
        sorted_dict = sorted(fitness_dict.items(), key=lambda x: x[1])
        self.population = [c[0] for c in sorted_dict]
