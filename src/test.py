from genotypes import CellularAutomaton1D, Genotype
import numpy as np
import matplotlib.pyplot as plt
import utils

size = 100
hood_size = 3
rule = utils.int_to_binary(90, 2**hood_size)
state = np.random.randint(0, 2, dtype='i1', size=size)
ca = CellularAutomaton1D(rule=rule, size=size, hood_size=hood_size)
ca.encode_staring_state(np.array([1,0,1,0]))
ca.run_time_evolution(size)

plt.imshow(ca.get_history(), cmap='gray')
plt.show()

#TODO asyncronous CA (50% of cells are updated)

