import copy

from genotypes import *
from utils.utils import *
from src.generation import Generation
from src import config


def random_bitarray(length: int):
    return np.random.randint(0, 2, length, dtype='u4')


def get_parent_index(parents: np.ndarray):
    # norm_parents = parents/sum(parents)
    norm_parents = (parents ** 2) / (sum(parents ** 2))
    check = np.random.rand()
    prob = 0

    for i, parent in enumerate(norm_parents):
        prob += parent
        if check < prob:
            return i


def mutation(parent: np.ndarray):
    if parent.ndim == 1:
        mutation_index = np.random.randint(parent.size)
        parent[mutation_index] = np.random.normal(-1, 1, size=1)
    else:
        mutation_index_row = np.random.randint(parent.shape[0])
        mutation_index_col = np.random.randint(parent.shape[1])
        parent[mutation_index_row, mutation_index_col] = np.random.normal(-1, 1, size=1)


def binary_mutation(parent: np.ndarray):
    if parent.ndim == 1:
        mutation_index = np.random.randint(parent.size)
        parent[mutation_index] = 1 - parent[mutation_index]

    else:
        mutation_index_row = np.random.randint(parent.shape[0])
        mutation_index_col = np.random.randint(parent.shape[1])
        parent[mutation_index_row, mutation_index_col] = 1 - parent[mutation_index_row, mutation_index_col]


# def crossover(parent_1: np.ndarray, parent_2: np.ndarray):
#     if parent_1.ndim == 1 and parent_2.ndim == 1:
#
#         rule_a, rule_b = one_point_crossover(parent_1, parent_2)
#         return rule_a, rule_b
#
#     else:
#         params1, params2 = one_point_crossover(parent_1, parent_2)
#         return params1, params2

    # https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_crossover


# def nn_crossover(parent_1: np.ndarray, parent_2: np.ndarray):
#     if parent_1.size > parent_2.size:
#         small_parent = parent_2
#         big_parent = parent_1
#     else:
#         small_parent = parent_1
#         big_parent = parent_2
#
#     x, y = small_parent.shape
#
#     row_start = np.random.randint(0, x)
#     row_end = np.random.randint(row_start + 1, x + 1)
#     col_start = np.random.randint(0, y)
#     col_end = np.random.randint(col_start + 1, y + 1)
#
#     small_gene = small_parent[row_start:row_end, col_start:col_end]
#     big_gene = big_parent[row_start:row_end, col_start:col_end]
#
#     print(small_gene)
#     print(big_gene)
#
#     small_parent[row_start:row_end, col_start:col_end] = big_gene
#     big_parent[row_start:row_end, col_start:col_end] = small_gene
#
#     return small_parent, big_parent


def one_point_crossover(parent_1: np.ndarray, parent_2: np.ndarray):
    cross_index = np.random.randint(len(parent_1))

    p1_gene_a = parent_1[:cross_index]
    p1_gene_b = parent_1[cross_index:]

    p2_gene_a = parent_2[:cross_index]
    p2_gene_b = parent_2[cross_index:]

    new_1 = np.concatenate((p1_gene_a, p2_gene_b))
   # new_2 = np.concatenate((p1_gene_b, p2_gene_a))
    new_2 = np.concatenate((p2_gene_a, p1_gene_b))
    return new_1, new_2


def uniform_crossover(parent_1: np.ndarray, parent_2: np.ndarray):

    A = np.copy(parent_1)
    B = np.copy(parent_2)

    mask = np.random.randint(0, 2, size=parent_1.shape, dtype='u4')

    A[mask == 1] = parent_2[mask == 1]
    B[mask == 1] = parent_1[mask == 1]

    return A, B


def generate_offspring_nn(parents: Generation):
    offspring = []
    parents_fitness = np.asarray([parent.get_fitness() for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()
    index = 1

    # for i in range(config.data['evolution']['number_of_elites']):
    #     parent1 = parents[-1 - i]
    #
    #     parent_number = parent1.candidate_number
    #     offspring.append(NeuralNetwork(parent_number, parent1.input_weights,
    #                                    parent1.hidden_layer_bias, parent1.output_weights))

    while len(offspring) < pop_size:
        parent1 = get_parent_index(parents_fitness)

        random_check = np.random.rand()

        generation_index = parents.generation_number + 1
        new_index = f"{generation_index}-{index}"

        if random_check < xover_ratio:
            parent2 = get_parent_index(parents_fitness)

            while parent1 == parent2:
                parent2 = get_parent_index(parents_fitness)

            new_input_weights_1, new_input_weights_2 = uniform_crossover(parents[parent1].input_weights,
                                                                         parents[parent2].input_weights)

            new_hidden_layer_bias_1, new_hidden_layer_bias_2 = uniform_crossover(parents[parent1].hidden_layer_bias,
                                                                                 parents[parent2].hidden_layer_bias)

            new_output_weights_1, new_output_weights_2 = uniform_crossover(parents[parent1].output_weights,
                                                                           parents[parent2].output_weights)

            offspring.append(NeuralNetwork(new_index,
                                           new_input_weights_1,
                                           new_hidden_layer_bias_1,
                                           new_output_weights_1))

            index += 1
            new_index = f"{generation_index}-{index}"

            offspring.append(NeuralNetwork(new_index,
                                           new_input_weights_2,
                                           new_hidden_layer_bias_2,
                                           new_output_weights_2))

        elif random_check < mutate_ratio + xover_ratio:
            mutate_nn(new_index, parents[parent1], offspring)

        else:
            offspring.append(NeuralNetwork(parents[parent1].candidate_number))

        index += 1

    return offspring[:config.data['evolution']['generation_size']]


def mutate_nn(index, parent: NeuralNetwork, offspring: List[NeuralNetwork]):
    selector = np.random.randint(0, 4)

    if selector == 0:
        mutate_input_weights(index, parent, offspring)

    elif selector == 1:
        mutate_hidden_layer_bias(index, parent, offspring)

    else:
        mutate_output_weights(index, parent, offspring)


def mutate_input_weights(index, parent, offspring):
    weights = np.copy(parent.input_weights)
    mutation(weights)
    offspring.append(NeuralNetwork(index,
                                   weights,
                                   parent.hidden_layer_bias,
                                   parent.output_weights))


def mutate_hidden_layer_bias(index, parent, offspring):
    bias = np.copy(parent.hidden_layer_bias)
    mutation(bias)
    offspring.append(NeuralNetwork(index,
                                   parent.input_weights,
                                   bias,
                                   parent.output_weights))


def mutate_output_weights(index, parent, offspring):
    weights = np.copy(parent.output_weights)
    mutation(weights)
    offspring.append(NeuralNetwork(index,
                                   parent.input_weights,
                                   parent.hidden_layer_bias,
                                   weights))


def generate_offspring_ca(parents: Generation):
    offspring = []
    parents_fitness = np.asarray([parent.get_fitness() for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()
    index = 1

    # for i in range(config.data['evolution']['number_of_elites']):
    #     parent1 = parents[-1 - i]
    #     offspring.append(CellularAutomaton1D(parent1.candidate_number, parent1.rule, parent1.size, parent1.steps))

    while len(offspring) < pop_size:

        parent1 = get_parent_index(parents_fitness)

        generation_index = parents.generation_number + 1
        new_index = f"{generation_index}-{index}"

        random_check = np.random.rand()

        if random_check < xover_ratio:
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
            mutate_ca(new_index, parents[parent1], offspring)

        else:
            offspring.append(CellularAutomaton1D(parents[parent1].candidate_number,
                                                 parents[parent1].rule,
                                                 parents[parent1].size,
                                                 parents[parent1].steps))
        index += 1

    return offspring[:config.data['evolution']['generation_size']]


def mutate_ca(index, parent: CellularAutomaton1D, offspring: List[CellularAutomaton1D]):
    selector = np.random.randint(0, 4)

    if selector == 0:
        ca_mutate_size(index, parent, offspring)

    elif selector == 1:
        ca_mutate_size(index, parent, offspring)

    else:
        ca_mutate_steps(index, parent, offspring)


def ca_mutate_rule(index, parent, offspring):
    rule = copy.deepcopy(parent.rule)
    binary_mutation(rule)
    offspring.append(CellularAutomaton1D(index,
                                         rule,
                                         parent.size,
                                         parent.steps))


def ca_mutate_size(index, parent, offspring):
    offspring.append(CellularAutomaton1D(index,
                                         parent.rule,
                                         np.random.randint(config.data['cellular_automata']['ca_size_low'],
                                                           config.data['cellular_automata']['ca_size_high']),
                                         parent.steps))


def ca_mutate_steps(index, parent, offspring):
    offspring.append(CellularAutomaton1D(index,
                                         parent.rule,
                                         parent.size,
                                         np.random.randint(config.data['cellular_automata']['ca_steps_low'],
                                                           config.data['cellular_automata']['ca_steps_high'])))
