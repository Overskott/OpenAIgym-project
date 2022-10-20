from abc import ABC, abstractmethod
from typing import List
import copy

import gym
import numpy as np

import config
import utils.utils as utils


class Genotype(ABC):

    @abstractmethod
    def run_time_evolution(self):
        pass

    @abstractmethod
    def get_history(self):
        pass

    @abstractmethod
    def get_fitness(self):
        pass

    @abstractmethod
    def set_fitness(self, fitness):
        pass

    @abstractmethod
    def clear_history(self):
        pass

    @abstractmethod
    def test_phenotype(self, environment: gym.Env, policy):
        pass


class CellularAutomaton1D(Genotype):

    def __init__(self,
                 rule: np.ndarray,
                 size: int = None,
                 configuration: np.ndarray = None):

        if size is None or size == 0:
            self.size = config.data['ca_size']
        else:
            self.size = size

        if configuration is None:
            self.configuration = np.zeros(self.size, dtype='i1')
        else:
            self.configuration = configuration

        self._rule = rule
        self.hood_size = config.data['ca_hood_size']
        self.history = []
        self.fitness = 0

    def __str__(self):
        return str(self.configuration)

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    @property
    def configuration(self) -> np.ndarray:
        return self._configuration

    @configuration.setter
    def configuration(self, state: np.ndarray):
        self._configuration = state
        self.size = len(state)

    @property
    def rule(self) -> np.ndarray:
        return self._rule

    @rule.setter
    def rule(self, rule: np.ndarray):
        self._rule = rule

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, size: int):
        self._size = size

    def encode_staring_state(self, observations: np.ndarray):
        self.configuration = np.zeros(self.size, dtype='i1')
        for i in range(len(observations)):
            self.configuration[(self.size >> 1) + (i - len(observations))] = observations[i]

    def encode_observables(self, observables):

        new_state = np.zeros(self.size)
        gap = int(config.data['ca_size'] / len(observables))

        for i, observation in enumerate(observables):
            if i == 0:
                b_array = utils.observable_to_binary_array(observation, -4.8, 4.8)
            elif i == 2:
                b_array = utils.observable_to_binary_array(observation, -0.418, 0.418)
            else:
                b_array = utils.observable_to_binary_array(observation, 10, 10)

            new_state[i * gap:i * gap + 10] = b_array

        self.configuration = new_state

    def test_phenotype(self, environment: gym.Env, policy):
        score = 0
        max_steps = 500
        observation, _ = environment.reset()

        for i in range(max_steps):

            action = policy(observation, self)  # User-defined policy function
            observation, reward, terminated, truncated, _ = environment.step(action)
            score += reward

            if terminated:
                break
            elif truncated:
                break

        self.set_fitness(score)

        environment.close()

    def run_time_evolution(self):
        for i in range(config.data['ca_steps']):
            self.history.append(self.configuration)
            self.configuration_time_step()

    def configuration_time_step(self):
        new_state = copy.deepcopy(self.configuration)

        for i in range(self.size):
            new_state[i] = self.cell_time_step(i)

        self.configuration = new_state

    def cell_time_step(self, cell_index, boundary_condition='Periodic'):
        if boundary_condition == 'Periodic':
            neighborhood = self.__get_cell_neighbourhood(cell_index)

            cell_new_value = self.__apply_rule(neighborhood)
        elif boundary_condition == 'Fixed':
            pass
        else:
            pass

        return cell_new_value

    def __apply_rule(self, neighborhood):
        hood_binary_value = int(neighborhood, 2)
        new_cell_value = int(self.rule[hood_binary_value])

        return new_cell_value

    def __get_cell_neighbourhood(self, cell_index: int) -> str:
        neighborhood = ""

        ca_array = np.concatenate((self.configuration[-self.hood_size:],
                                   self.configuration,
                                   self.configuration[:self.hood_size]))

        start_cell = cell_index + self.hood_size - 1
        end_cell = start_cell + self.hood_size

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

