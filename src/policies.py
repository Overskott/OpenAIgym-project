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
    state = CA.observables_to_binary(observation)[1:]
    action = CA.increment_state(state, rule)

    return action

def spread_out(observations, rule, size=20):
    obser = CA.observables_to_binary(observations)

    new_state = [0 for _ in range(size)]

    for i in range(len(obser)):

        new_state[i*(size >> len(obser))] = obser[i]

    new_state = [str(x) for x in new_state]
    
    for run in range(size):
        new_state = CA.increment_ca(new_state, rule)

    return new_state[size >> 2]




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
