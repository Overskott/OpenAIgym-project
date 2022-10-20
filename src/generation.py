from typing import List

from src import policies
from genotypes import Genotype, CellularAutomaton1D
import numpy as np
import config
from utils.utils import *


class Population(object):

    def __init__(self, id_number):
        self.parents: List[Genotype] = []
        self.population: List[Genotype] = []
        self.id = id_number
        self.generation_size = config.data['generation_size']

    def random_rule(self):
        return np.random.randint(0, 2, 2 ** config.data['ca_hood_size'], dtype='i1')

    def initialize_population(self):
        self.population = []

        for i in range(self.generation_size):
            rule = self.random_rule()
            phenotype = CellularAutomaton1D(rule)
            self.population.append(phenotype)

    def select_parents(self) -> List[Genotype]:
        self.sort_children_by_fitness()
        return self.population[:self.generation_size >> 1]

    def sort_children_by_fitness(self):
        fitness_dict = {f: f.get_fitness() for f in self.population}
        sorted_dict = sorted(fitness_dict.items(), key=lambda x: x[1])
        self.population = [c[0] for c in sorted_dict]

    def get_next_generation(self):
        next_generation = []
        if len(self.parents) == 0:
            self.initialize_population()
        else:
            for i in range(0, len(self.parents)-2):

                parent_1 = self.parents[i-1]
                parent_2 = self.parents[i]

                rule1, rule2 = self.rule_crossover(parent_1, parent_2)
                child1 = CellularAutomaton1D(rule1)
                child2 = CellularAutomaton1D(rule2)

                self.population.append(child1)
                self.population.append(child2)

            next_generation.append(self.parents[-4])
            next_generation.append(self.parents[-3])
            next_generation.append(self.parents[-2])
            next_generation.append(self.parents[-1])

        return next_generation

    def rule_crossover(self, parent_1: Genotype, parent_2: Genotype):

        hood_size = config.data['ca_hood_size']
        rule_length = 2 ** hood_size

        cross_index = np.random.randint(rule_length / 3, rule_length - rule_length / 3)

        p1_gene_a = parent_1.rule[:cross_index]
        p1_gene_b = parent_1.rule[cross_index:]
        p2_gene_a = parent_2.rule[:cross_index]
        p2_gene_b = parent_2.rule[cross_index:]

        rule_a = np.concatenate((p1_gene_a, p2_gene_b))
        rule_b = np.concatenate((p1_gene_b, p2_gene_a))

        return rule_a, rule_b

