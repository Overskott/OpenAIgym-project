from src.genotypes import *
from src.utils import *
import numpy as np
from src.policies import *
import matplotlib.pyplot as plt
size = 200
conf = np.zeros(size)
mid = size//2
#conf[mid-2] = 0
#conf[mid-1] = 1
conf[mid] = 1
#conf[mid+1] = 0
print(np.size(np.array([[1,2],[3,4],[5,6]])))
#
# fit = np.array([236.0,
# 243.0,
# 134.0,
# 158.0,
# 459.0,
# 500.0,
# 115.0,
# 194.0,
# 273.0,
# 86.0,
# 144.0,
# 164.0,
# 283.0,
# 101.0,
# 184.0,
# 134.0,
# 146.0,
# 236.0,
# 171.0,
# 353.0,
# 122.0,
# 244.0,
# 123.0,
# 264.0,
# 138.0,
# 153.0,
# 97.0,
# 163.0,
# 193.0,
# 133.0,
# 90.0,
# 172.0,
# 98.0,
# 160.0,
# 149.0,
# 201.0,
# 148.0,
# 136.0,
# 158.0,
# 168.0,
# 270.0,
# 146.0,
# 164.0,
# 225.0,
# 153.0,
# 196.0])
#
# fit2 = np.array([12.0,
# 314.0,
# 194.0,
# 300.0,
# 148.0,
# 13.0,
# 140.0,
# 304.0,
# 329.0,
# 132.0,
# 136.0,
# 279.0,
# 179.0,
# 365.0,
# 126.0,
# 31.0,
# 156.0,
# 166.0,
# 168.0,
# 160.0,
# 275.0,
# 170.0,
# 22.0,
# 170.0,
# 202.0,
# 138.0,
# 14.0,
# 157.0,
# 143.0,
# 289.0,
# 10.0,
# 12.0,
# 242.0,
# 310.0,
# 12.0,
# 128.0,
# 14.0,
# 123.0,
# 124.0])
#
# print(len(fit2))
#
# print(sum(fit2)/len(fit2))
# print(max(fit2))
# print(min(fit2))