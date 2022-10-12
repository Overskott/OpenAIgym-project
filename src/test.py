from genotypes import CellularAutomaton1D
import numpy as np
import matplotlib.pyplot as plt

size = 500
state = np.random.randint(0, 2, dtype='i1', size=size)

ca = CellularAutomaton1D(state=state, iterations=size)
steps = size

print(f"rule: {ca.rule}")
plot_list = []

ca.run()

print(ca.get_history)
plt.imshow(ca.get_history(), cmap='gray')
plt.show()

