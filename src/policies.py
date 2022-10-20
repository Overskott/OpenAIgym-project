
from genotypes import CellularAutomaton1D
from utils import utils
import numpy as np


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


def simple_ca(observation, model: CellularAutomaton1D):
    ca = model
    obs = utils.observables_to_binary(observation)
    ca.encode_staring_state(obs)
    ca.run_time_evolution()
    action = voting_result(ca.configuration)

    return action


def wide_encoding(observations, model: CellularAutomaton1D):
    ca = model

    new_state = np.zeros(ca.size)
    for i, observation in enumerate(observations):
        b_array = utils.observable_to_binary_array(observation, -4.8, 4.8)
        if i == 2:
            b_array = utils.observable_to_binary_array(observation, -0.418, 0.418)
        new_state[i*25:i*25+10] = b_array
    ca.configuration = new_state
    ca.run_time_evolution()
    action = voting_result(ca.configuration)

    return action


def voting_result(array: np.ndarray):
    result = int(np.round(sum(array)/len(array)))
    return result


