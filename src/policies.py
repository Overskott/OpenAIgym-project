
from genotypes import CellularAutomaton1D
import utils
import numpy as np
import matplotlib.pyplot as plt


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
    obs = utils.observables_to_binary(observation)[1:]
    configuration = np.zeros(model.size)

    for i, value in enumerate(obs):
        configuration[i*len(obs)] = value

    ca = model
    ca.configuration = configuration
    ca.run_time_evolution(ca.size)
    action = voting_result(ca.configuration)

    return action


def voting_result(array: np.ndarray):
    result = int(np.round(sum(array)/len(array)))
    print(result)
    return result


def voting_rule(observations, rule) -> int:
    configuration = utils.observables_to_binary(observations)
    ca = CellularAutomaton1D(configuration, rule,)
    ca.run_time_evolution(4)
    action = int(np.round(sum(configuration/len(configuration))))

    return action


def spread_out(observations, rule, size=40):
    obs = CA.observables_to_binary(observations)
    state = np.zeros(size, dtype='i4')
    distance = int(size/len(obs))

    for i in range(len(obs)):
        state[i*distance] = obs[i]

    ca = CA.CellularAutomaton1D(configuration=state, rule=rule, iterations=size)

    ca.run_time_evolution()
    sum = 0

    for value in ca.configuration:
        sum += value

    if sum < int(size/2):
        action = 0
    else:
        action = 1
    return action

def funnel(observation, rule, size=10):
    pass
