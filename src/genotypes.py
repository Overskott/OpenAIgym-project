from abc import ABC, abstractmethod
from typing import List
import copy
import gym
import numpy as np

import src.config as config
import src.utils as utils


class Genotype(ABC):
    """ An abstract class for creating genotypes for the OpenAI pole-cart problem.

        This class is adding needed functions and parameters for us with the policy.py and evolution.py modules.

        Attributes:
            fitness (int) : Variable for storing the Genotype's fitness.
        """

    def __init__(self):
        """Initialize the Genotype with 0 (lowest) fitness."""
        self.fitness = 0

    def fitness(self) -> int:
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
    """ Class for instantiating phenotypes as 1-dimensional cellular automata.

        This class contains all the parameters and functions for creating and running a 1-dimensional CA to evolve
        a solution to the OpenAi pole cart challenge.

        Attributes:
            history (List[CellularAutomata1D]): A list of the CA evolution for each step.
            candidate_number (str): Phenotype identifier.
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
                 configuration: np.ndarray = None,
                 hood_size: int = None):

        """ Initializing an instance of the CellularAutomata1D class. Randomizes the
            attributes if no attribute value is passed (except candidate_number).

            Args:
                candidate_number (str): String to identify phenotype in an evolution.
                rule (numpy.ndarray): The rule to be applied by the cellular automaton
                size (int): The number of cells in th 1D CA.
                steps (int): Number of time steps the CA evolution will take before terminating.
                configuration (numpy.ndarray): The current state of the CA

        """
        super().__init__()
        self.history = []
        self.candidate_number = candidate_number

        if hood_size is None:
            self.hood_size = config.data['cellular_automata']['ca_hood_size']
        else:
            self.hood_size = hood_size

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
            self.rule = np.random.randint(0, 2, 2 ** self.hood_size, dtype='u4')
        else:
            self.rule = rule

    def __str__(self):
        """ Prints the binary array representing the CA's current configuration and parameters"""
        return f"Cellular Automata ID: {self.candidate_number}, size: {self.size}, steps: {self.steps}, " \
               f"Neighborhood size: {self.hood_size}, Rule: {self.rule}\n" \
               f"\nParameters: \n" \
               f"{config.data['evolution']}\n" \
               f"{config.data['cellular_automata']}\n"

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

    def get_history(self) -> List[np.ndarray]:
        return self.history

    def clear_history(self):
        """ Clears the stored history list"""
        self.history = []

    def encode_observations(self, observations):
        """ Most efficient observations encoding scheme.

            Encodes the observations into the configuration based on size.

            Args:
                observations (numpy.ndarray) : The observations to be encoded.
            """
        size = config.data['cellular_automata']['observation_encoding_size']
        new_state = np.zeros(self.size, dtype='u4')
        gap = round(self.size / len(observations))

        for i, observation in enumerate(observations):
            if i == 0:
                b_array = utils.observable_to_binary_array(observation, -4.8, 4.8)
            elif i == 2:
                # b_array = utils.observable_to_binary_array(observation, -0.418, 0.418)
                b_array = utils.observable_to_binary_array(observation, -0.2095, 0.2095)
            else:
                b_array = utils.observable_to_binary_array(observation, -100, 100)

            new_state[i * gap: i * gap + size] = b_array

        self.configuration = new_state

    def find_phenotype_fitness(self, environment: gym.Env, policy):
        """ Tests the instance in the cartpole environment and updates its fitness score. Uses the
            given policy to determine the action to take. Config parameter 'test_rounds' determines
            the number of tests to run. Fitness is the average over all test runs.

            Args:
                environment (gym.Env): Gym environment for the instance to be tested in.
                policy (Func): The policy to be tested.
        """
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

    def run_time_evolution(self):
        """ The complete time evolution of the CA. Updates configuration."""
        self.clear_history()
        for i in range(self.steps):
            self.history.append(self.configuration)
            self.__configuration_time_step()

    def __configuration_time_step(self):
        """Updates the whole CA one time step. Updates configuration."""
        new_state = np.zeros(self.size, dtype='u4')

        for i in range(self.size):
            new_state[i] = self.__cell_time_step(i)

        self.configuration = new_state

    def __cell_time_step(self, cell_index) -> int:
        """ Updates given cell in the CA on time step.

            Args:
                cell_index (int): The index in the array to be updated.

            Returns:
                (int): The new cell value {0,1}
        """
        neighborhood = self.__get_cell_neighbourhood(cell_index)
        cell_new_value = self.__state_transition_function(neighborhood)

        return cell_new_value

    def __state_transition_function(self, neighborhood: str) -> int:
        """ Checks the given neighborhood against the rule attribute and returns its value

            Args:
                neighborhood (str): Binary representation of the cell neighborhood.

            Returns:
                (int): The cell value after applying its rule
        """
        hood_binary_value = int(neighborhood, 2)
        new_cell_value = int(self.rule[hood_binary_value])

        return new_cell_value

    def __get_cell_neighbourhood(self, cell_index: int) -> str:
        """ Find the neighborhood of the cell at given index. This is a Periodic boundary space.

            Args:
                cell_index (int): Index of cell to find neighborhood of.

            Returns:
                (str): A binary string representation of the neighborhood.
        """
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

    @DeprecationWarning
    def naive_encoding(self, observations: np.ndarray):
        """ A simple and inefficient way of encoding gym observations into the CA.

            The observations given are encoded in the middle of the ca.

            NOTE! This function is created for testing and research purpose, and will
            not yield the best results.

            Args:
                observations (numpy.ndarray): An array with the four environment observations
                in the OpenAI CartPole environment. The observations must be in binary value and
                in this order:

                [Cart position, Cart velocity, Pole angle, Pole angular velocity]
        """
        self.configuration = np.zeros(self.size, dtype='u4')
        for i in range(len(observations)):
            self.configuration[(self.size >> 1) + (i - len(observations))] = observations[i]


class NeuralNetwork(Genotype):
    """ Class for instantiating phenotypes as artificial neural networks.

        This class contains all the parameters and functions for creating and running a neural network to evolve
        a solution to the OpenAi pole cart challenge. n is the number of nodes in the hidden layer.

        Attributes:
            candidate_number (str): Phenotype identifier.
            input_layer (numpy.ndarray): The input to the NN (dim(1, 4))
            input_weights (numpy.ndarray): Weights between input and hidden layer (dim(4, n)).
            hidden_bias (numpy. ndarray): The hidden layer bias (dim(1, n)).
            output_weights (numpy.ndarray): weights between hidden layer and output layer (dim(n, 1)).
        """

    def __init__(self,
                 candidate_number: str,
                 input_weights: np.ndarray = None,
                 input_bias: np.ndarray = None,
                 hidden_bias: np.ndarray = None,
                 output_weights: np.ndarray = None):
        """ Initializing a NeuralNetwork instance. Randomizes the attributes if none is given (except from candidate_number)

            Args:
                candidate_number (str): Phenotype identifier.
                input_weights (numpy.ndarray): Weights between input and hidden layer (dim(4, n)).
                hidden_bias (numpy. ndarray): The hidden layer bias (dim(1,n)).
                output_weights (numpy.ndarray): weights between hidden layer and output layer (dim(n,1)).
        """
        super().__init__()

        if input_weights is None:
            self.input_weights = np.random.normal(-1, 1, (4, config.data['neural_network']['hidden_layer_size']))
        else:
            self.input_weights = input_weights

        if input_bias is None:
            # self.input_bias = np.random.normal(-1, 1, (1, 4))
            self.input_bias = np.zeros((1, 4))
        else:
            self.input_bias = input_bias

        if hidden_bias is None:
            # self.hidden_bias = np.random.normal(-1, 1, (1, config.data['neural_network']['hidden_layer_size']))
            self.hidden_bias = np.zeros((1, config.data['neural_network']['hidden_layer_size']))
        else:
            self.hidden_bias = hidden_bias

        if output_weights is None:
            self.output_weights = np.random.normal(-1, 1, (config.data['neural_network']['hidden_layer_size'], 1))
        else:
            self.output_weights = output_weights

        self.candidate_number = candidate_number
        self.input_values = np.zeros((1, 4))

    def __str__(self):
        """ Prints the arrays representing the NN's current configuration and parameters"""
        return f"NeuralNetwork ID: {self.candidate_number}, Fitness: {self.fitness}\n" \
               f"\nParameters: {config.data['evolution']}\n" \
               f"{config.data['neural_network']}\n" \
               f"\nInput Weights:\n{utils.array_to_input_string(self.input_weights)}\n" \
               f"\nInput Bias:\n{utils.array_to_input_string(self.input_bias)}" \
               f"\nHidden Bias:\n{utils.array_to_input_string(self.hidden_bias)}\n" \
               f"\nOutput_weights:\n{utils.array_to_input_string(self.output_weights)}"

    @property
    def input_weights(self) -> np.ndarray:
        return self._input_weights

    @input_weights.setter
    def input_weights(self, weights: np.ndarray):
        self._input_weights = weights

    @property
    def input_bias(self) -> np.ndarray:
        return self._input_bias

    @input_bias.setter
    def input_bias(self, input_bias):
        self._input_bias = input_bias

    @property
    def hidden_bias(self) -> np.ndarray:
        return self._hidden_layer_bias

    @hidden_bias.setter
    def hidden_bias(self, bias: np.ndarray):
        self._hidden_layer_bias = bias

    @property
    def output_weights(self) -> np.ndarray:
        return self._output_weights

    @output_weights.setter
    def output_weights(self, weights: np.ndarray):
        self._output_weights = weights

    def set_input_values(self, input_array: np.ndarray):
        self.input_values = copy.deepcopy(input_array)

    def calculate_output_value(self) -> float:
        """ Calculates and returns the output of the NN given the input.

            Returns:
                (float): The NN output value
        """
        input_layer = self.input_values + self.input_bias
        hidden_layer = np.dot(input_layer, self.input_weights) + self.hidden_bias
        output_value = np.dot(hidden_layer, self.output_weights)

        return np.float(output_value)

    def find_phenotype_fitness(self, environment: gym.Env, policy):
        """ Tests the instance in the cartpole environment and updates its fitness score. Uses the
            given policy to determine the action to take. Config parameter 'test_rounds' determines
            the number of tests to run. Fitness is the average over all test runs.

                    Args:
                        environment (gym.Env): Gym environment for the instance to be tested in.
                        policy (Func): The policy to be tested.
                """
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
