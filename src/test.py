import numpy as np

from evolution import *

A = np.random.random((4, 5))
B = np.random.random((4, 5))



C, D = ca_crossover(A, B)


#
print(f"A: {A}")
print(f"B: {B}")
print(f"C: {C}")
print(f"D: {D}")



