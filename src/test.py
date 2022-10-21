import numpy as np
import copy
from utils.utils import *
import evolution
import matplotlib.pyplot as plt
import genotypes

bit1 = np.array([1,3,2,6])
bit2 = np.array([0,0,0,0])
bit3 = np.array([1,0,1,0])

p1 = genotypes.CellularAutomaton1D(evolution.random_bitarray(2 ** 4))
p2 = genotypes.CellularAutomaton1D(evolution.random_bitarray(2 ** 4))
p3 = genotypes.CellularAutomaton1D(evolution.random_bitarray(2 ** 4))

parents = [p1, p2, p3]

print(bit1/sum(bit1))