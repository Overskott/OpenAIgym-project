import numpy as np
import src.config as config
import os
from pathlib import Path
import matplotlib.pyplot as plt


def binary_to_int(array: np.ndarray):
    """Converting a numpy array with binary values to decimal integer"""
    binary_string = ''.join(f"{binary}" for binary in array)
    number = int(binary_string, 2)

    return number


def int_to_binary(value: int, length: int) -> np.ndarray:
    """ Converting a decimal integer to a binary number as a numpy array with binary values"""
    binary_string = format(value, f"0{length}b")
    binary_array = np.array([int(x) for x in binary_string])

    return binary_array


def observables_to_binary(observations):
    """ Takes cartpole observations and encode them into a binary array to use as initial condition for a CA.

        Args:
            observations (numpy.nd.array): The cartpole environment observations
            [position, velocity, angle, angular_momentum]

        Returns:
              (numpy.ndarray): THe observations encoded into a CA initial state.
        """
    state = np.zeros(len(observations), dtype='u4')
    for i, observable in enumerate(observations):
        if i == 0 and np.abs(observable) > 0.5:
            state[i] = 1
        elif observable >= 0:
            state[i] = 1
        else:
            state[i] = 0
    return state


def observable_to_binary_array(observable: float, low, high,
                               array_size=config.data['cellular_automata']['observation_encoding_size']):

    values = np.linspace(low, high, array_size+1)
    binary_array = np.zeros(config.data['cellular_automata']['observation_encoding_size'], dtype='u4')

    for i in range(array_size):
        if observable <= values[i + 1]:
            binary_array[i] = 1
            break

    return binary_array


def get_date_and_time():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    return now.strftime("%d.%m.%Y-%Hh%Mm%Ss")


def array_to_input_string(array):
    output_string = '['
    for i, row in enumerate(array):
        output_string += '['

        for j, element in enumerate(row):
            if j < len(row)-1:
                output_string += str(element)
                output_string += ', '
            else:
                output_string += str(element)

        if i < len(array)-1:
            output_string += '],\n'
        else:
            output_string += ']'

    output_string += ']'

    return output_string


def save_nn_results(text, figure=None):
    time_finished = get_date_and_time()
    folder_name = f"NN results {time_finished}"
    current_path = Path.cwd()
    parent_dir = current_path.parents[0]

    result_folder_path = parent_dir / 'results'

    create_run_result_folder(folder_name, result_folder_path)
    write_to_file(text, folder_name, result_folder_path, folder_name)

    if figure is not None:
        plt.savefig(result_folder_path / folder_name / f'{figure}.png')


def save_ca_results(text, figure=None, figure2=None):
    time_finished = get_date_and_time()
    folder_name = f"CA results {time_finished}"
    current_path = Path.cwd()
    parent_dir = current_path.parents[0]

    result_folder_path = parent_dir / 'results'

    create_run_result_folder(folder_name, result_folder_path)
    write_to_file(text, folder_name, result_folder_path, folder_name)

    if figure is not None:
        plt.figure(figure)
        plt.savefig(result_folder_path / folder_name / f'{figure}.png')

    if figure2 is not None:
        plt.figure(figure2)
        plt.savefig(result_folder_path / folder_name / f'{figure2}.png')


def save_figure(figure, fig_name):
    time_finished = get_date_and_time()
    folder_name = f"CA results {time_finished}"
    current_path = Path.cwd()
    parent_dir = current_path.parents[0]

    result_folder_path = parent_dir / 'results'
    plt.figure(figure)
    plt.savefig(result_folder_path / folder_name / f'{fig_name}.png')


def create_run_result_folder(folder_name, folder_path):

    if folder_path.is_dir():
        os.makedirs(folder_path / folder_name)
    else:
        os.makedirs(folder_path)
        print('Directory doesn\'t exist')
        print(f'Creating {folder_path}')

        os.makedirs(folder_path / folder_name)


def write_to_file(text, file_name, folder_dir, folder_name):

    file_name = f"{file_name}.txt"
    result_folder_path = folder_dir / folder_name

    with open(result_folder_path.joinpath(file_name), 'w') as f:
        f.write(text)


