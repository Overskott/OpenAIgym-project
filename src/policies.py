import stat

import matplotlib.pyplot as plt

import genotypes
import random
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


def simple_ca(observation, rule):
    obs = CA.observables_to_binary(observation)[1:]
    ca = CA.CellularAutomaton1D(state=obs)
    ca.increment_ca()
    action = ca.state[1]

    return action

def spread_out(observations, rule, size=40):
    obs = CA.observables_to_binary(observations)
    state = np.zeros(size, dtype='i4')
    distance = int(size/len(obs))

    for i in range(len(obs)):
        state[i*distance] = obs[i]

    ca = CA.CellularAutomaton1D(state=state, rule=rule, iterations=size)

    ca.run()

    sum = 0

    for value in ca.state:
        sum += value


    if sum < int(size/2):
        action = 0
    else:
        action = 1
    return action

def funnel(observation, rule, size=10):
    pass

def voter_control(observation):
    state = CA.observables_to_binary(observation)
    sum = 0
    for i in state:
        sum += int(i)
    print(sum)
    if sum/4 == 1/2:
        return random.randint(0, 1)
    elif sum/4 > 1/2:
        return 0
    else:
        return 1
