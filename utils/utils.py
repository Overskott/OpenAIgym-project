import random
import numpy as np


def binary_to_int(array: np.ndarray):
    """Converting a numpy array with binary values to decimal integer"""
    binary_string = ''.join(f"{binary}" for binary in array)
    number = int(binary_string, 2)

    return number


def int_to_binary(value: int, length: int) -> np.ndarray:
    """ Converting a decimal integer to a numpy array with binary values"""
    binary_string = format(value, f"0{length}b")
    binary_array = np.array([int(x) for x in binary_string])

    return binary_array


def get_generation_sorted_by_fitness(generation):
    fitness_dict = {f: f.get_fitness() for f in generation}
    sorted_fitness_dict = sorted(fitness_dict.items(), key=lambda x: x[1])
    return [t[0] for t in sorted_fitness_dict]


def observables_to_binary(observables):
    """Observables = [position, velocity, angle, angular_momentum]"""
    state = np.zeros(len(observables), dtype='i1')
    for i, observable in enumerate(observables):
        if i == 0 and np.abs(observable) > 0.5:
            state[i] = 1
        elif observable >= 0:
            state[i] = 1
        else:
            state[i] = 0
    return state


def observable_to_binary_array(observable: float, low, high, array_size=10):
    """range {-4.8, 4.8}"""
    discrete = np.linspace(low, high, array_size+1)
    binary_array = np.zeros(array_size, dtype='i1')

    for i in range(array_size):
        if observable <= discrete[i + 1]:
            binary_array[i] = 1
            break
    return binary_array

