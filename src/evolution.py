import copy

from genotypes import *
from utils.utils import *
from src.generation import Generation
from src import config


def random_bitarray(length: int):
    return np.random.randint(0, 2, length, dtype='u4')

def get_parent_index(parents: np.ndarray):
    #norm_parents = parents/sum(parents)
    norm_parents = (parents**2)/(sum(parents**2))
    check = np.random.rand()
    prob = 0

    for i, parent in enumerate(norm_parents):
        prob += parent
        if check < prob:
            return i

    return norm_parents[-1].candidate_number

def mutation(bitarray: np.ndarray):
    mutation_index = np.random.randint(bitarray.size)

    bitarray[mutation_index] = 1 - bitarray[mutation_index]


def crossover(parent_1: np.ndarray, parent_2: np.ndarray):

    bitarray_length = len(parent_1)

    if bitarray_length != len(parent_2):
        raise ValueError("Parent rules must have the same length")

    cross_index = np.random.randint(bitarray_length)

    p1_gene_a = parent_1[:cross_index]
    p1_gene_b = parent_1[cross_index:]

    p2_gene_a = parent_2[:cross_index]
    p2_gene_b = parent_2[cross_index:]

    rule_a = np.concatenate((p1_gene_a, p2_gene_b))
    rule_b = np.concatenate((p1_gene_b, p2_gene_a))

    return rule_a, rule_b


def generate_offspring_nn(parents: Generation):
    offspring = []
    parents_fitness = np.asarray([parent.get_fitness() for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()
    index = 1
    # print(f"parents_fitness: {parents_fitness}")
    for i in range(config.data['evolution']['number_of_elites']):

        parent1 = parents[-1-i]

        parent_number = parent1.candidate_number
        offspring.append(NeuralNetwork(parent_number, parent1.input_weights,
                                       parent1.hidden_layer_bias, parent1.output_weights))

        index += 1

    while len(offspring) < pop_size:
        generation_index = parents.generation_number + 1
        new_index = f"{generation_index}-{index}"
        offspring.append(NeuralNetwork(new_index))

        index += 1

    return offspring[:config.data['evolution']['generation_size']]


def generate_offspring_ca(parents: Generation):
    offspring = []
    parents_fitness = np.asarray([parent.get_fitness() for parent in parents.population])
    pop_size = config.data['evolution']['generation_size']
    xover_ratio = config.data['evolution']['crossover_rate']
    mutate_ratio = config.data['evolution']['mutation_rate']
    parents.sort_population_by_fitness()

    for i in range(config.data['evolution']['number_of_elites']):

        parent1 = parents[-i]
        offspring.append(CellularAutomaton1D(parent1.rule, parent1.size, parent1.steps))

    while len(offspring) < pop_size:

        parent1 = get_parent_index(parents_fitness)

        random_check = np.random.rand()

        if random_check < xover_ratio:
            parent2 = get_parent_index(parents_fitness)

            while parent1 == parent2:
                parent2 = get_parent_index(parents_fitness)

            rule1, rule2 = crossover(parents[parent1].rule, parents[parent2].rule)
            # print(f" rule1: {binary_to_int(rule1)}, rule2: {binary_to_int(rule2)}")

            if binary_to_int(rule1) == binary_to_int(rule2):
                mutation(rule2)

            offspring.append(CellularAutomaton1D(rule1, parents[parent1].size, parents[parent1].steps))
            offspring.append(CellularAutomaton1D(rule2, parents[parent2].size, parents[parent2].steps))

            # print("Crossover {} x {}".format(parent1, parent2))

        elif random_check < mutate_ratio + xover_ratio:
            rule = copy.deepcopy(parents[parent1].rule)
            mutation(rule)

            selector = np.random.randint(0, 4)
            if selector == 0:
                offspring.append(CellularAutomaton1D(rule,
                                                     parents[parent1].size,
                                                     parents[parent1].steps))
                # print("Mutation rule {}".format(parent1))
            elif selector == 1:
                offspring.append(CellularAutomaton1D(parents[parent1].rule,
                                                     np.random.randint(config.data['cellular_automata']['ca_size_low'],
                                                                       config.data['cellular_automata']['ca_size_high']),
                                                     parents[parent1].steps))
                # print("Mutation size {}".format(parent1))
            else:
                offspring.append(CellularAutomaton1D(parents[parent1].rule,
                                                     parents[parent1].size,
                                                     np.random.randint(config.data['cellular_automata']['ca_steps_low'],
                                                                       config.data['cellular_automata']['ca_steps_high'])))
                # print("Mutation steps {}".format(parent1))
        else:
            offspring.append(CellularAutomaton1D(parents[parent1].rule,
                                                 parents[parent1].size,
                                                 parents[parent1].steps))
            # offspring.append(CellularAutomaton1D(random_bitarray(2 ** config.data['ca_hood_size'])))
            # print("Elitism {}".format(parents.population[-1].get_fitness()))

    return offspring[:config.data['evolution']['generation_size']]


