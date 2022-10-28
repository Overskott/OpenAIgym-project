from genotypes import *
from utils import *
import numpy as np
from policies import *
import matplotlib.pyplot as plt
# size = 20
# conf = np.zeros(size)
# mid = size//2
# conf[mid-2] = 0
# conf[mid-1] = 1
# conf[mid] = 1
# conf[mid+1] = 0
#
# while True:
#     ca = CellularAutomaton1D('test', hood_size=3, size=size, configuration=conf, steps=size)
#
#
#     ca.run_time_evolution()
#
#
#     plt.imshow(ca.history, cmap='gray', interpolation='nearest')
#     plt.show()


x = np.linspace(1, 500, 1000)

yy = x/sum(x)
y = x**2/sum(x**2)

plt.plot(x, yy, label='y = x/sum(x)')
plt.plot(x, y)
plt.show()