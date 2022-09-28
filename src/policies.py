import stat

import matplotlib.pyplot as plt

import CA
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
    ca = CA.OneDimCA(state=obs)
    ca.increment_ca()
    action = ca.state[1]

    return action

def spread_out(observations, rule, size=20):
    obs = CA.observables_to_binary(observations)
    state = np.zeros(size, dtype='i4')
    distance = int(size/len(obs))

    for i in range(len(obs)):
        state[i*distance] = obs[i]

    ca = CA.OneDimCA(state=state, rule=rule)

    plot_list = []
    for i in range(size):
        plot_list.append(ca.state)
        ca.increment_ca()

    sum = 0
    for value in ca.state:
        sum += value

    if sum/len(ca.state) < 0.5:
        action = 0
    else:
        action = 1

    plt.imshow(plot_list, cmap='gray')
    plt.show()
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
