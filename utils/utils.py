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


def observables_to_binary(observables):
    """Observables = [position, velocity, angle, angular_momentum]"""
    state = np.zeros(len(observables), dtype='u4')
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
    values = np.linspace(low, high, array_size+1)
    binary_array = np.zeros(array_size, dtype='u4')

    for i in range(array_size):
        if observable <= values[i + 1]:
            binary_array[i] = 1
            break

    return binary_array


