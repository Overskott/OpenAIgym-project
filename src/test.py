from genotypes import CellularAutomaton1D
import numpy as np
import matplotlib.pyplot as plt
from utils import utils

space = np.linspace(-4.8, 4.8, 11)
cells = np.zeros(10, dtype='i1')

obs = 4

for i in range(10):
    if obs < space[i+1]:
        cells[i] = 1
        break

print(space)
print(cells)


#TODO asyncronous CA (50% of cells are updated)

