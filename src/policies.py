
from genotypes import CellularAutomaton1D
from utils import utils
import numpy as np
import config


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

    model.encode_observables(observations)
    model.run_time_evolution()
    action = voting_result(model.configuration)

    return action


def random_generation(observations, model: CellularAutomaton1D):
    ca = model
    ca.encode_observables(observations)
    ca.run_time_evolution()
    action = voting_result(ca.configuration)

    return action


def voting_result(array: np.ndarray):
    result = int(np.round(sum(array)/len(array)))
    return result


