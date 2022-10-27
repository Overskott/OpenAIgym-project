import numpy as np

from evolution import *

A = np.random.random((4, 3))
B = np.random.random((4, 3))

C, D = uniform_crossover(A, B)

print(A)
print(B)
print(C)
print(D)
