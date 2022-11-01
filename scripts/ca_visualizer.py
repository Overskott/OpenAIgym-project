from src.genotypes import *
from src.utils import *
import numpy as np
from src.policies import *
import matplotlib.pyplot as plt

size = 300
conf = np.zeros(size)
mid = size//2
rule=153
#conf[mid-2] = 0
#conf[mid-1] = 1
conf[mid] = 1
#conf[mid+1] = 0

while True:
    ca = CellularAutomaton1D('test', hood_size=3, size=size, configuration=conf, steps=size, rule=int_to_binary(rule, 2**3))

    print(binary_to_int(ca.rule))
    ca.run_time_evolution()

    plt.imshow(ca.history, cmap='gray')
    plt.show()
