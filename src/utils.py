import random
import numpy as np

def random_ca_rule(self):
    return random.randint(0, 255)

def observables_to_binary(observables):
    """Observables = [position, velocity, angle, angular_momentum]"""
    state = np.zeros(len(observables))
    for i, observable in enumerate(observables):
        if observable >= 0:
            state[i] = 1
        else:
            state[i] = 0
    return state
