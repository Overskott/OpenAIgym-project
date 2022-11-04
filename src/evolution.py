from typing import List

from src.genotypes import CellularAutomaton1D, NeuralNetwork
from src.generation import Generation
from src.utils import binary_to_int
import src.config as config

import numpy as np


def random_binary_array(length: int):
    """ Returns an array of size given as parameter with random values of {0, 1}"""
    return np.random.randint(0, 2, length, dtype='u4')


def get_parent_index(parent_fitness_array: np.ndarray):
    """ Returns the index of the parent to be selected from the parent fitness array

        Args:
            parent_fitness_array: An array of fitness values of the parents

        Returns:
            The index of the parent to be selected
    """
    norm_fitness = normalize_array(parent_fitness_array)
    return fitness_proportionate_selector(norm_fitness)


def normalize_array(input_array: np.ndarray) -> np.ndarray:
    """ Normalizes the input array to a sum of 1 """
    return input_array/np.sum(input_array)


@DeprecationWarning
def normalize_array_squared(input_array: np.ndarray) -> np.ndarray:
    """ Normalizes the square of the input array to a sum of 1 """
    return input_array**2/np.sum(input_array**2)


def fitness_proportionate_selector(selection_array: np.ndarray) -> int:
    """ Returns the index of the parent to be selected by the fitness proportionate selection method """
    check = np.random.rand()
    prob = 0

    for i, candidate in enumerate(selection_array):
        prob += candidate
        if check < prob:
            return i


def mutation(parent: np.ndarray):
    """
    Mutates the given parent array with float values. If argument is 1-dimensional, one value is randomly changed.
    A multidimensional array will 1/3 ov the values be randomly changed. The mutation is done in place
    """

    if parent.ndim == 1:
        mutation_index = np.random.randint(parent.size)
        parent[mutation_index] = np.random.normal(-1, 1, size=1)
    else:
        for i in range(np.size(parent) // 3):
            # Mutate 1/3 of the matrix entries
            mutation_index_row = np.random.randint(parent.shape[0])
            mutation_index_col = np.random.randint(parent.shape[1])
            parent[mutation_index_row, mutation_index_col] = np.random.normal(-1, 1, size=1)


def binary_mutation(parent: np.ndarray):
    """
    Mutates the given parent array with binary values. If argument is 1-dimensional, one value is randomly changed.
    A multidimensional array will 1/3 ov the values be randomly changed. The mutation is done in place
    """

    if parent.ndim == 1:
        mutation_index = np.random.randint(parent.size)
        parent[mutation_index] = 1 - parent[mutation_index]

    else:
        for i in range(np.size(parent)//3):
            # Mutate 1/3 of the matrix entries
            mutation_index_row = np.random.randint(parent.shape[0])
            mutation_index_col = np.random.randint(parent.shape[1])
            parent[mutation_index_row, mutation_index_col] = 1 - parent[mutation_index_row, mutation_index_col]


def one_point_crossover(parent_1: np.ndarray, parent_2: np.ndarray):
    """ Performs a one point crossover on the given parents by one-point crossover.

        Args:
            parent_1: The first parent
            parent_2: The second parent

        Returns:
            The two offspring genes
    """
    cross_index = np.random.randint(len(parent_1))

    p1_gene_a = parent_1[:cross_index]
    p1_gene_b = parent_1[cross_index:]

    p2_gene_a = parent_2[:cross_index]
    p2_gene_b = parent_2[cross_index:]

    new_1 = np.concatenate((p1_gene_a, p2_gene_b))
    new_2 = np.concatenate((p2_gene_a, p1_gene_b))
    return new_1, new_2


def uniform_crossover(parent_1: np.ndarray, parent_2: np.ndarray):
    """ Performs a crossover on the given parents by uniform crossover.

        Args:
            parent_1: The first parent
            parent_2: The second parent

        Returns:
            The two offspring genes
    """
    A = np.copy(parent_1)
    B = np.copy(parent_2)

    mask = np.random.randint(0, 2, size=parent_1.shape, dtype='u4')

    A[mask == 1] = parent_2[mask == 1]
    B[mask == 1] = parent_1[mask == 1]

    return A, B


def generate_offspring_nn(parents: Generation):
    """ Generates offspring from the given NN parents by mutation, elitism and crossover """
    offspring = []
    parents_fitness = np.asarray([parent.fitness for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()
    index = 1

    while len(offspring) < pop_size:
        parent1 = get_parent_index(parents_fitness)

        random_check = np.random.rand()

        generation_index = parents.generation_number + 1
        new_index = f"{generation_index}-{index}"

        if random_check < xover_ratio:
            # Crossover
            parent2 = get_parent_index(parents_fitness)

            while parent1 == parent2:
                parent2 = get_parent_index(parents_fitness)

            new_input_weights_1, new_input_weights_2 = uniform_crossover(parents[parent1].input_weights,
                                                                         parents[parent2].input_weights)

            new_input_bias_1, new_input_bias_2 = uniform_crossover(parents[parent1].input_bias,
                                                                   parents[parent2].input_bias)

            new_hidden_layer_bias_1, new_hidden_layer_bias_2 = uniform_crossover(parents[parent1].hidden_bias,
                                                                                 parents[parent2].hidden_bias)

            new_output_weights_1, new_output_weights_2 = uniform_crossover(parents[parent1].output_weights,
                                                                           parents[parent2].output_weights)

            offspring.append(NeuralNetwork(new_index,
                                           new_input_weights_1,
                                           new_input_bias_1,
                                           new_hidden_layer_bias_1,
                                           new_output_weights_1))

            index += 1
            new_index = f"{generation_index}-{index}"

            offspring.append(NeuralNetwork(new_index,
                                           new_input_weights_2,
                                           new_input_bias_2,
                                           new_hidden_layer_bias_2,
                                           new_output_weights_2))

        elif random_check < mutate_ratio + xover_ratio:
            # Mutate
            mutate_nn(new_index, parents[parent1], offspring)

        else:
            # Elitism
            offspring.append(NeuralNetwork(parents[parent1].candidate_number))

        index += 1

    return offspring[:config.data['evolution']['generation_size']]


def mutate_nn(index, parent: NeuralNetwork, offspring: List[NeuralNetwork]):
    """ Selecting the NN attribute to be mutated """
    selector = np.random.randint(0, 4)

    if selector == 0:
        mutate_input_weights(index, parent, offspring)

    elif selector == 1:
        mutate_hidden_bias(index, parent, offspring)

    elif selector == 2:
        mutate_input_bias(index, parent, offspring)

    else:
        mutate_output_weights(index, parent, offspring)


def mutate_input_weights(index, parent, offspring):
    """ Mutates the input weights of the given parent and appends the new NN to the offspring list """
    weights = np.copy(parent.input_weights)
    mutation(weights)
    offspring.append(NeuralNetwork(index,
                                   weights,
                                   parent.input_bias,
                                   parent.hidden_bias,
                                   parent.output_weights))


def mutate_hidden_bias(index, parent, offspring):
    """ Mutates the hidden bias of the given parent and appends the new NN to the offspring list """
    bias = np.copy(parent.hidden_bias)
    mutation(bias)
    offspring.append(NeuralNetwork(index,
                                   parent.input_weights,
                                   parent.input_bias,
                                   bias,
                                   parent.output_weights))


def mutate_output_weights(index, parent, offspring):
    """ Mutates the output weights of the given parent and appends the new NN to the offspring list """
    weights = np.copy(parent.output_weights)
    mutation(weights)
    offspring.append(NeuralNetwork(index,
                                   parent.input_weights,
                                   parent.input_bias,
                                   parent.hidden_bias,
                                   weights))


def mutate_input_bias(index, parent, offspring):
    """ Mutates the input bias of the given parent and appends the new NN to the offspring list """
    bias = np.copy(parent.input_bias)
    mutation(bias)
    offspring.append(NeuralNetwork(index,
                                   parent.input_weights,
                                   bias,
                                   parent.hidden_bias,
                                   parent.output_weights))


def generate_offspring_ca(parents: Generation):
    """ Generates offspring from the given CA parents by mutation, elitism and crossover """
    offspring = []
    parents_fitness = np.asarray([parent.fitness for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()
    index = 1

    while len(offspring) < pop_size:

        parent1 = get_parent_index(parents_fitness)

        generation_index = parents.generation_number + 1
        new_index = f"{generation_index}-{index}"

        random_check = np.random.rand()

        if random_check < xover_ratio:
            # Crossover
            parent2 = get_parent_index(parents_fitness)

            while parent1 == parent2:
                parent2 = get_parent_index(parents_fitness)

            rule1, rule2 = one_point_crossover(parents[parent1].rule, parents[parent2].rule)

            if binary_to_int(rule1) == binary_to_int(rule2):
                binary_mutation(rule2)

            offspring.append(CellularAutomaton1D(new_index, rule1, parents[parent1].size, parents[parent1].steps))

            index += 1
            new_index = f"{generation_index}-{index}"

            offspring.append(CellularAutomaton1D(new_index, rule2, parents[parent2].size, parents[parent2].steps))

        elif random_check < mutate_ratio + xover_ratio:
            # Mutate
            mutate_ca(new_index, parents[parent1], offspring)

        else:
            # Elitism
            offspring.append(CellularAutomaton1D(parents[parent1].candidate_number,
                                                 parents[parent1].rule,
                                                 parents[parent1].size,
                                                 parents[parent1].steps))
        index += 1

    return offspring[:config.data['evolution']['generation_size']]


def mutate_ca(index, parent: CellularAutomaton1D, offspring: List[CellularAutomaton1D]):
    """ Selecting the CA attribute to be mutated """
    selector = np.random.randint(0, 4)

    if selector == 0:
        ca_mutate_size(index, parent, offspring)

    elif selector == 1:
        ca_mutate_size(index, parent, offspring)

    else:
        ca_mutate_steps(index, parent, offspring)


def ca_mutate_rule(index, parent, offspring):
    """ Mutates the rule of the given parent and appends the new CA to the offspring list """
    rule = np.copy(parent.rule)
    binary_mutation(rule)
    offspring.append(CellularAutomaton1D(index,
                                         rule,
                                         parent.size,
                                         parent.steps))


def ca_mutate_size(index, parent, offspring):
    """ Mutates the size of the given parent and appends the new CA to the offspring list """
    offspring.append(CellularAutomaton1D(index,
                                         parent.rule,
                                         np.random.randint(config.data['cellular_automata']['ca_size_low'],
                                                           config.data['cellular_automata']['ca_size_high']),
                                         parent.steps))


def ca_mutate_steps(index, parent, offspring):
    """ Mutates the steps of the given parent and appends the new CA to the offspring list """
    offspring.append(CellularAutomaton1D(index,
                                         parent.rule,
                                         parent.size,
                                         np.random.randint(config.data['cellular_automata']['ca_steps_low'],
                                                           config.data['cellular_automata']['ca_steps_high'])))
