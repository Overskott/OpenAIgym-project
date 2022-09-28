import random
import numpy as np


def random_ca_rule(self):
    return random.randint(0, 255)



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

class OneDimCA(object):

    def __init__(self, size, rule):
        self.state = self.State(size)
        self.rule = rule

    class State(np.ndarray):

        def __new__(cls, size):
            random_nparray = np.random.randint(0, 2, size)
            return random_nparray

        def __init__(self, size):
            self.array_size = size
            # self.state = self.generate_random_state()
            super().__init__(self)

        def __str__(self):
            return str(self)

        def generate_random_state(self):
            state = np.random.randint(0, 2, dtype='i1', size=self.size)
            return state

        def cell_time_step(self, cell_index, rule, boundary_condition='None'):

            rule_binary_list = [x for x in format(rule, '08b')]
            neighborhood = ""

            if boundary_condition == 'Periodic':
                pass
            elif boundary_condition == 'Fixed':
                pass
            elif boundary_condition == 'Cut-off':
                pass
            else:
                neighborhood = self.get_three_cell_neighbourhood()

                nh_binary_value = int(neighborhood, 2)
                cell_new_value = int(rule_binary_list[nh_binary_value])

            self[cell_index] = cell_new_value

        def get_three_cell_neighbourhood(self, cell_index):

            neighborhood = ""
            neighborhood += str(np.mod(cell_index - (3 >> 1), self.size))
            neighborhood += str(np.mod(cell_index, self.size))
            neighborhood += str(np.mod(cell_index + (3 >> 1), self.size))

            return neighborhood

        def get_five_cell_neighbourhood(self, cell_index):

            neighborhood = ""
            neighborhood += str(np.mod(cell_index - (5 >> 1), self.size))
            neighborhood += str(np.mod(cell_index - (5 >> 2), self.size))
            neighborhood += str(np.mod(cell_index, self.size))
            neighborhood += str(np.mod(cell_index + (5 >> 2), self.size))
            neighborhood += str(np.mod(cell_index + (5 >> 1), self.size))

            return neighborhood

        def increment_ca(self, rule):
            for i in range(self.size):
                self.cell_time_step(i, rule)





