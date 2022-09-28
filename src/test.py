from CA import OneDimCA
import numpy as np
import matplotlib.pyplot as plt

print(np.random.randint(0, 2, dtype='i1', size=10))
size = 20
ca = OneDimCA()
steps = size

print(f"rule: {ca.rule}")
plot_list = []
for i in range(steps):
    plot_list.append(ca.state)
    ca.increment_ca()


plt.imshow(plot_list, cmap='gray')
plt.show()

