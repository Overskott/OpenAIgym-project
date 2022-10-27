from abc import ABC, abstractmethod
from typing import List
import copy

import gym
import numpy as np

import config
import utils.utils as utils


class Genotype(ABC):
    """ An abstract class for creating genotypes for the OpenAI pole-cart problem.

        This class is adding needed functions and parameters for us with the policy.py and evolution.py modules.

        Attributes:
            fitness (int) : Variable for storing the Genotype's fitness.
        """

    def __init__(self):
        """Initialize the Genotype with 0 (lowest) fitness."""
        self.fitness = 0

    def get_fitness(self) -> int:
        """Returns the fitness of the Genotype."""
        return self.fitness

    def set_fitness(self, fitness: int):
        """Sets the value of fitness to argument value"""
        self.fitness = fitness

    @abstractmethod
    def find_phenotype_fitness(self, environment: gym.Env, policy):
        """Abstract method to run the pole cart environment with the genotype as and return the fitness

        Args:
            environment (gym.Env): The OpenAi gym environment the genotype will be tested in.
            Only CartPole-v1 is supported.
            policy: A policy defined in the policies.py module. Different policies are supported by different
            genotpes.
        """
        pass


class CellularAutomaton1D(Genotype):
    """ Class for instantiating genotypes as 1-dimensional cellular automata.

        This class contains all the parameters and functions for creating and running a 1-dimensional CA to evolve
        a solution to the OpenAi pole cart challenge.

        Attributes:
            history (List[CellularAutomata1D]): A list of the CA evolution for each step.
            candidate_number (str): First number to add.
            rule (numpy.ndarray): Array containing the rules for the CA to follow.
            size (int): The size, or width of the CA. Can also be interpreted as number of cells.
            steps (int): Number of steps the CA will run.
            configuration (numpy.ndarray): The status of the CA i.e. which cells are 1 and which are 0.
        """
    def __init__(self,
                 candidate_number: str,
                 rule: np.ndarray = None,
                 size: int = None,
                 steps: int = None,
                 configuration: np.ndarray = None):
        """ Initializing an instance of the CellularAutomata1D class. Randomizes most of the
            attributes if no value is passed.

            Args:
                candidate_number (str):
                rule (numpy.ndarray):
                size (int):
                steps (int):
                configuration (numpy.ndarray):

        """
        super().__init__()
        self.history = []
        self.candidate_number = candidate_number
        self.hood_size = config.data['cellular_automata']['ca_hood_size']

        if size is None or size == 0:
            self.size = np.random.randint(config.data['cellular_automata']['ca_size_low'],
                                          config.data['cellular_automata']['ca_size_high'])
        else:
            self.size = size

        if steps is None:
            self.steps = np.random.randint(config.data['cellular_automata']['ca_steps_low'],
                                           config.data['cellular_automata']['ca_steps_high'])
        else:
            self.steps = steps

        if configuration is None:
            self.configuration = np.zeros(self.size, dtype='u4')
        else:
            self.configuration = copy.deepcopy(configuration)

        if rule is None:
            self.rule = np.random.randint(0, 2, 2**self.hood_size, dtype='u4')
        else:
            self.rule = rule

    def __str__(self):
        return str(self.configuration)

    @property
    def configuration(self) -> np.ndarray:
        return self._configuration

    @configuration.setter
    def configuration(self, state: np.ndarray):
        self._configuration = copy.deepcopy(state)
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

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

    def encode_staring_state(self, observations: np.ndarray):
        self.configuration = np.zeros(self.size, dtype='i1')
        for i in range(len(observations)):
            self.configuration[(self.size >> 1) + (i - len(observations))] = observations[i]

    def encode_observables(self, observables):

        size = config.data['cellular_automata']['observation_encoding_size']
        new_state = np.zeros(self.size, dtype='u4')
        gap = round(self.size / len(observables))

        for i, observation in enumerate(observables):
            if i == 0:
                b_array = utils.observable_to_binary_array(observation, -4.8, 4.8)
            elif i == 2:
                # b_array = utils.observable_to_binary_array(observation, -0.418, 0.418) # -.2095, .2095
                b_array = utils.observable_to_binary_array(observation, -0.2095, 0.2095)
            else:
                b_array = utils.observable_to_binary_array(observation, -100, 100)

            new_state[i * gap: i * gap + size] = b_array

        self.configuration = new_state

    def find_phenotype_fitness(self, environment: gym.Env, policy):
        score = 0
        repeats = config.data['evolution']['test_rounds']
        for step in range(repeats):
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

            self.set_fitness(np.round(score/repeats))

        environment.close()

    def run_time_evolution(self):
        self.clear_history()
        for i in range(self.steps):
            self.history.append(self.configuration)
            self.configuration_time_step()

    def configuration_time_step(self):
        new_state = np.zeros(self.size, dtype='u4')

        for i in range(self.size):
            new_state[i] = self.cell_time_step(i)

        self.configuration = new_state

    def cell_time_step(self, cell_index):

        neighborhood = self.__get_cell_neighbourhood(cell_index)
        cell_new_value = self.__apply_rule(neighborhood)

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

    # def generate_gene(self) -> List[any]:
    #     gene_list = [self.configuration, self.rule, self.hood_size]
    #
    #     return gene_list
    #
    # def parse_gene(self, gene):
    #     self.configuration = gene[0]
    #     self.rule = gene[0]
    #     self.hood_size = gene[2]
    #     pass
    #
    # def showcase_phenotype(self, environment, policy):
    #     env = environment
    #     score = 0
    #     model = self
    #     observation, _ = env.reset()
    #
    #     for _ in range(500):
    #         action = policy(observation, model)  # User-defined policy function
    #         observation, reward, terminated, truncated, _ = env.step(action)
    #
    #         score += reward
    #
    #         if terminated:
    #             env.reset()
    #             print(f"Run terminated with score {score}")
    #             break
    #         elif truncated:
    #             env.reset()
    #             print(f"Run truncated with score {score}")
    #             break
    #
    #     env.close()


class NeuralNetwork(Genotype):
    """Add up two integer numbers.

        This function simply wraps the ``+`` operator, and does not
        do anything interesting, except for illustrating what
        the docstring of a very simple function looks like.

        Args:
            num1 (int) : First number to add.
            num2 (int) : Second number to add.

        Returns:
            The sum of ``num1`` and ``num2``.

        Raises:
            AnyError: If anything bad happens.
        """
    def __init__(self,
                 candidate_number: str,
                 input_weights: np.ndarray = None,
                 hidden_layer_bias: np.ndarray = None,
                 output_weights: np.ndarray = None):

        super().__init__()

        if input_weights is None:
            self.input_weights = np.random.normal(-1, 1, (4, config.data['neural_network']['hidden_layer_size']))
        else:
            self.input_weights = input_weights

        if hidden_layer_bias is None:
            self.hidden_layer_bias = np.random.normal(-1, 1, (1, config.data['neural_network']['hidden_layer_size']))
        else:
            self.hidden_layer_bias = hidden_layer_bias

        if output_weights is None:
            self.output_weights = np.random.normal(-1, 1, (config.data['neural_network']['hidden_layer_size'], 1))
        else:
            self.output_weights = output_weights

        self.candidate_number = candidate_number
        self.input_layer = np.zeros((1, 4))
        self.fitness = 0

    def __str__(self):
        pass

    @property
    def input_weights(self) -> np.ndarray:
        return self._input_weights

    @input_weights.setter
    def input_weights(self, weights: np.ndarray):
        self._input_weights = weights

    @property
    def hidden_layer_bias(self) -> np.ndarray:
        return self._hidden_layer_bias

    @hidden_layer_bias.setter
    def hidden_layer_bias(self, bias: np.ndarray):
        self._hidden_layer_bias = bias

    @property
    def output_weights(self) -> np.ndarray:
        return self._output_weights

    @output_weights.setter
    def output_weights(self, weights: np.ndarray):
        self._output_weights = weights

    def set_input_values(self, input_array: np.ndarray):
        self.input_layer = copy.deepcopy(input_array)

    def calculate_output_value(self) -> float:

        hidden_layer = np.dot(self.input_layer, self.input_weights) + self.hidden_layer_bias
        output_value = np.dot(hidden_layer, self.output_weights)

        return np.float(output_value)

    def find_phenotype_fitness(self, environment: gym.Env, policy):
        score = 0
        repeats = config.data['evolution']['test_rounds']
        for step in range(repeats):
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

            self.set_fitness(np.round(score / repeats))

        # environment.close()

