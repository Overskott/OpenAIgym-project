import random
import numpy as np


class Genotype(object):

    def __int__(self):
        pass


class CellularAutomaton1D(Genotype):

    def __init__(self, state=None,
                 rule=None,
                 iterations=10,
                 stride=1,
                 neighbourhood=3):

        if state is None:
            self.state = np.random.randint(0, 2, dtype='i4', size=10)
        else:
            self.state = state

        if rule is None:
            self.rule = [x for x in format(90, '08b')]
        else:
            self.rule = [x for x in format(rule, '08b')]

        self.size = len(self.state)
        self.iterations = iterations
        self.history = []

    def __str__(self):
        return str(self.state)

    def run(self):
        for i in range(self.iterations):
            self.increment_ca()
            self.history.append(self.state)

    def cell_time_step(self, cell_index, boundary_condition='Periodic'):
        neighborhood = ""
        if boundary_condition == 'Periodic':
            neighborhood = self.get_three_cell_neighbourhood(cell_index)
            nh_binary_value = int(neighborhood, 2)
            cell_new_value = int(self.rule[nh_binary_value])
        elif boundary_condition == 'Fixed':
            pass
        else:
            pass

        return cell_new_value

    def get_three_cell_neighbourhood(self, cell_index: int) -> str:

        neighborhood = ""
        neighborhood += str(self.state[cell_index - 1])
        neighborhood += str(self.state[cell_index])
        neighborhood += str(self.state[(cell_index + 1) % self.size])

        return neighborhood

    def get_five_cell_neighbourhood(self, cell_index):

        neighborhood = ""
        neighborhood += str(np.mod(cell_index - (5 >> 1), self.size))
        neighborhood += str(np.mod(cell_index - (5 >> 2), self.size))
        neighborhood += str(np.mod(cell_index, self.size))
        neighborhood += str(np.mod(cell_index + (5 >> 2), self.size))
        neighborhood += str(np.mod(cell_index + (5 >> 1), self.size))

        return neighborhood

    def get_history(self):
        return self.history

    def increment_ca(self):
        new_state = np.zeros(self.size, dtype='i4')
        for i in range(self.size):
            new_state[i] = self.cell_time_step(i)

        self.state = new_state




