from genotypes import CellularAutomaton1D, NeuralNetwork
import config
import numpy as np


class Generation(object):
    """


    """
    def __init__(self, genotype, id_number, population=None):

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
        return np.asarray([phenotype.get_fitness() for phenotype in self.population])

    def sort_population_by_fitness(self):
        fitness_dict = {f: f.get_fitness() for f in self.population}
        sorted_dict = sorted(fitness_dict.items(), key=lambda x: x[1])
        self.population = [c[0] for c in sorted_dict]
