import src.utils as utils
import numpy as np  
from src.genotypes import CellularAutomaton1D, NeuralNetwork


def ca_wide_encoding(observations, model: CellularAutomaton1D):
    """ An encoding policy of the cartpole observations in a binary CA. Uses
    the CellularAutomata1D.encode_observations() scheme.

        Args:
            observation (numpy.ndarray): observations from the polecart environment.
            model (CellularAutomaton1D): The CA to encode.

        Returns:
            (int): Action based on the majority value in CA after completing all time steps.
    """
    model.encode_observations(observations)
    model.run_time_evolution()
    action = voting_result(model.configuration)

    return action


def voting_result(array: np.ndarray):
    """ Returns the majority value in a binary array. {1} is return if there is no majority.

        Args:
            array (numpy.ndarray): The binary array to be evaluated.

        Returns:
            integer {0, 1}, the majority value in the bit array
    """
    result = int(np.round(sum(array)/len(array)))
    return result


def nn_basic_encoding(observation, model: NeuralNetwork):
    """ Encoding of cartpole observations as input layer in NN.

        Args:
            observation (numpy.ndarray): observations from the cartpole environment.
            model (NeuralNetwork): The NN to encode input layer.

        Returns:
            (int): Action from Sigmoid activation function of the NN output.
    """

    nn = model
    obs = utils.observables_to_binary(observation).reshape(1, 4)
    nn.set_input_values(obs)
    result = nn.calculate_output_value()
    action = sigmoid(result)

    return int(action)


def sigmoid(value: float) -> int:
    """ Returns the rounded value of the sigmoid function"""
    return np.round(1/(1+np.exp(-value)))


@DeprecationWarning
def right_control():
    """ Test policy that just moves the cart to the right."""
    return 1


@DeprecationWarning
def random_control():
    """ Test policy with random actions"""
    import random

    action = random.randint(0, 1)
    return action


@DeprecationWarning
def naive_control(observation):
    """ Test policy that takes action based on pole angle

        Args:
            observation (numpy.ndarray): observations from the polecart environment.

        Returns:
            (int): 1 if angle is positive, else 0
    """
    if observation[2] >= 0:
        return 1
    else:
        return 0


@DeprecationWarning
def simple_encoding(observation, model: CellularAutomaton1D) -> int:
    """ A narrow encoding policy of the cartpole observations in a binary CA.

        Args:
            observation (numpy.ndarray): observations from the polecart environment.
            model (CellularAutomaton1D): The CA to encode.

        Returns:
            (int): Action based on the majority value in CA after completing all time steps.
    """

    ca = model
    obs = utils.observables_to_binary(observation)
    ca.naive_encoding(obs)
    ca.run_time_evolution()
    action = voting_result(ca.configuration)

    return action
