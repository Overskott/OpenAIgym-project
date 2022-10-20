from typing import List

from src import policies
from genotypes import Genotype, CellularAutomaton1D
import numpy as np
from utils.config_parser import get_config_file
from utils.utils import *


class Population(object):

    def __init__(self):
        self.population = []
        self.id = id

        self.parameters = get_config_file()['parameters']

    def random_rule(self):
        return np.random.randint(0, 2, 2 ** self.parameters['cellular_automata']['hood_size'], dtype='i1')

    def initialize_population(self):
        self.population = []
        gen_size = self.parameters['evolution']['generation_size']

        for i in range(gen_size):
            rule = self.random_rule()
            phenotype = CellularAutomaton1D(rule)
            self.population.append(phenotype)

    def select_parents(self) -> List[Genotype]:
        generation_sorted = get_generation_sorted_by_fitness(self.population)
        best_individuals = generation_sorted[len(generation_sorted) >> 1:]
        print(f"Best ind: {[x.get_fitness() for x in best_individuals]}")
        return best_individuals

    def create_next_generation(self, parents: List):

        parents = parents
        new_generation = []
        data = get_config_file()['parameters']
        gen_size = data['evolution']['generation_size']

        for i in range(0, len(parents)-2):

            parent_1 = parents[i-1]
            parent_2 = parents[i]

            rule1, rule2 = self.rule_crossover(parent_1, parent_2)
            phenotype = CellularAutomaton1D(rule1)
            new_generation.append(phenotype)
            phenotype = CellularAutomaton1D(rule2)
            new_generation.append(phenotype)

        new_generation.append(parents[-4])
        new_generation.append(parents[-3])
        new_generation.append(parents[-2])
        new_generation.append(parents[-1])


        self.population = new_generation

    def rule_crossover(self, parent_1: CellularAutomaton1D, parent_2: CellularAutomaton1D):

        hood_size = self.parameters['cellular_automata']['hood_size']
        rule_length = 2 ** hood_size

        cross_index = np.random.randint(rule_length / 3, rule_length - rule_length / 3)

        p1_gene_a = parent_1.rule[:cross_index]
        p1_gene_b = parent_1.rule[cross_index:]
        p2_gene_a = parent_2.rule[:cross_index]
        p2_gene_b = parent_2.rule[cross_index:]

        rule_a = np.concatenate((p1_gene_a, p2_gene_b))
        rule_b = np.concatenate((p1_gene_b, p2_gene_a))

        return rule_a, rule_b

