import random
from typing import List
import copy
import numpy as np
from abc import ABC, abstractmethod


class Genotype(ABC):

    @abstractmethod
    def run_time_evolution(self, steps):
        pass

    @abstractmethod
    def get_history(self):
        pass


class CellularAutomaton1D(Genotype):

    def __init__(self, configuration: np.ndarray,
                 rule: int,
                 hood_size=3):

        self.hood_size = hood_size
        self.configuration = configuration
        self._rule = rule
        self._size = len(self.configuration)
        self.history = []

    def __str__(self):
        return str(self.configuration)

    def __format_rule(self, rule: int) -> List[str]:
        rule_string = [x for x in format(rule, f"0{2 ** self.hood_size}b")]

        return rule_string

    def get_history(self):
        return self.history

    @property
    def configuration(self) -> np.ndarray:
        return self._configuration

    @configuration.setter
    def configuration(self, config: np.ndarray):
        self._configuration = config
        self.size = len(self.configuration)

    @property
    def rule(self):
        return self._rule

    @rule.setter
    def rule(self, rule: int):
        self._rule = rule

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    def cell_time_step(self, cell_index, boundary_condition='Periodic'):

        if boundary_condition == 'Periodic':
            neighborhood = self.__get_cell_neighbourhood(cell_index)
            cell_new_value = self.__apply_rule(neighborhood)
        elif boundary_condition == 'Fixed':
            pass
        else:
            pass

        return cell_new_value

    def configuration_time_step(self):
        new_state = copy.copy(self.configuration)

        for i in range(self._size):
            new_state[i] = self.cell_time_step(i)

        self.configuration = new_state

    def run_time_evolution(self, steps):
        for i in range(steps):
            self.configuration_time_step()
            self.history.append(self.configuration)

    def __apply_rule(self, neighborhood):

        hood_binary_value = int(neighborhood, 2)
        binary_rule = self.__format_rule(self.rule)
        new_cell_value = int(binary_rule[hood_binary_value])

        return new_cell_value

    def __get_cell_neighbourhood(self, cell_index: int) -> str:
        neighborhood = ""

        ca_array = np.concatenate((self.configuration[-self.hood_size:],
                                   self.configuration,
                                   self.configuration[:self.hood_size]))

        start_cell = cell_index + self.hood_size
        end_cell = (start_cell + self.hood_size)

        hood_range = ca_array[start_cell: end_cell]

        for cell in hood_range:
            neighborhood += str(int(cell))

        return neighborhood

    def generate_gene(self) -> List[any]:
        gene_list = [self.configuration, self.rule, self.hood_size]

        return gene_list

    def parse_gene(self, gene):
        self.configuration = gene[0]
        self.rule = gene[0]
        self.hood_size = gene[2]
        pass

