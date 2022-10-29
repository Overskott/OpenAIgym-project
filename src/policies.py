import utils as utils
import numpy as np

from genotypes import CellularAutomaton1D, NeuralNetwork


def right_control(observation):
    return 1


def random_control(observation):
    import random

    action = random.randint(0, 1)
    return action


def naive_control(observation):
    if observation[2] >= 0:
        return 1
    else:
        return 0


def simple_encoding(observation, model: CellularAutomaton1D):
    ca = model
    obs = utils.observables_to_binary(observation)
    ca.naive_encoding(obs)
    ca.run_time_evolution()
    action = voting_result(ca.configuration)

    return action


def wide_encoding(observations, model: CellularAutomaton1D):

    model.encode_observations(observations)
    model.run_time_evolution()
    action = voting_result(model.configuration)

    return action


def voting_result(array: np.ndarray):
    result = int(np.round(sum(array)/len(array)))
    return result


def nn_basic_encoding(observation, model: NeuralNetwork):
    nn = model
    obs = utils.observables_to_binary(observation).reshape(1, 4)
    nn.set_input_values(obs)
    result = nn.calculate_output_value()
    action = activation_func(result)

    return int(action)


def activation_func(value):
    return np.round(sigmoid(value))


def sigmoid(value: float) -> float:
    return 1/(1+np.exp(-value))
