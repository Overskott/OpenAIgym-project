from genotypes import CellularAutomaton1D, Genotype
import numpy as np
import matplotlib.pyplot as plt
from utils import binary_to_int

size = 10
state = np.random.randint(0, 2, dtype='i1', size=size)
ca = CellularAutomaton1D(configuration=state, rule=90, hood_size=3)

ca.run_time_evolution(size)

print(binary_to_int(state))
plt.imshow(ca.get_history(), cmap='gray')
plt.show()

#TODO asyncronous CA (50% of cells are updated)

