from typing import List

from src import policies
from genotypes import Genotype, CellularAutomaton1D
import numpy as np
from utils.config_parser import get_config_file
from utils.utils import *

def initialize_population():

    data = get_config_file()['parameters']

    gen_size = data['evolution']['generation_size']
    generation = []

    for i in range(gen_size):
        rule = np.random.randint(0,
                                 2,
                                 2 ** data['cellular_automata']['hood_size'],
                                 dtype='i1')
        phenotype = CellularAutomaton1D(rule)
        generation.append(phenotype)

    return generation


def select_parents(generation: List[Genotype]) -> List[Genotype]:
    generation_sorted = get_generation_sorted_by_fitness(generation)
    best_individuals = generation_sorted[len(generation_sorted) >> 1:]

    return best_individuals


def create_next_generation(parents: List) -> List[Genotype]:

    parents = parents
    new_generation = parents
    data = get_config_file()['parameters']
    gen_size = data['evolution']['generation_size']

    for i in range(0, gen_size, 2):
        parent_1 = parents[i]
        parent_2 = parents[i+1]

        rule1, rule2 = rule_crossover(parent_1, parent_2)
        phenotype = CellularAutomaton1D(rule1)
        new_generation.append(phenotype)
        phenotype = CellularAutomaton1D(rule2)
        new_generation.append(phenotype)

    return new_generation


def rule_crossover(parent_1: CellularAutomaton1D, parent_2: CellularAutomaton1D):

    data = get_config_file()['parameters']

    hood_size = data['cellular_automata']['hood_size']
    rule_length = 2**hood_size

    cross_index = np.random.randint(rule_length/3, rule_length-rule_length/3)

    p1_gene_a = parent_1.rule[:cross_index]
    p1_gene_b = parent_1.rule[cross_index:]
    p2_gene_a = parent_2.rule[:cross_index]
    p2_gene_b = parent_2.rule[cross_index:]

    rule_a = np.concatenate((p1_gene_a, p2_gene_b))
    rule_b = np.concatenate((p1_gene_b, p2_gene_a))

    return rule_a, rule_b


def test_genotype(environment, genotype: Genotype, policy):
    score = 0
    max_steps = 500
    observation, _ = environment.reset()

    for i in range(max_steps):

        action = policy(observation, genotype)  # User-defined policy function
        observation, reward, terminated, truncated, _ = environment.step(action)
        score += reward

        if terminated:
            break
        elif truncated:
            break

    genotype.set_fitness(score)

    environment.close()
