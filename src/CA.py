import random
import numpy as np


def initialize_random_ca(size):
    state = [random.randint(0, 1) for x in range(size)]
    return state


def random_ca_rule():
    return random.randint(0, 255)


def increment_ca(state, rule):

    new_ca = []
    for i, value in enumerate(state):
        neighborhood = ""
        neighborhood += str(state[i - 1])
        neighborhood += str(state[i])
        neighborhood += str(state[np.mod(len(state) - i, len(state))])

        new_ca.append(increment_state(neighborhood, rule))

    return new_ca


def increment_state(neighborhood, rule):

    rule_list = [x for x in format(rule, '08b')]
    new_state = (int(rule_list[int(neighborhood, 2)]))

    return new_state


def observables_to_binary(observables):
    position = observables[0]
    velocity = observables[1]
    angle = observables[2]
    angle_mom = observables[3]

    state = ""
    if position >= 0:
        state += '1'
    else:
        state += '0'

    if velocity >= 0:
        state += '1'
    else:
        state += '0'

    if angle >= 0:
        state += '1'
    else:
        state += '0'

    if angle_mom >= 0:
        state += '1'
    else:
        state += '0'

    return state


def print_state(state):
    string = ""
    for bit in state:
        if bit == 1:
            string += "@"
        else:
            string += "_"
    print(string)



