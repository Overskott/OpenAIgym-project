from src.genotypes import *
from src.utils import *
import numpy as np
from src.policies import *
import matplotlib.pyplot as plt

size = 200
conf = np.zeros(size)
mid = size // 2
# conf[mid-2] = 0
# conf[mid-1] = 1
conf[mid] = 1
# conf[mid+1] = 0
print(np.size(np.array([[1, 2], [3, 4], [5, 6]])))
#
hidden_4 = np.array(
    [168.0, 184.0, 245.0, 411.0, 258.0, 245.0, 327.0, 192.0, 335.0, 214.0, 249.0, 179.0, 197.0, 185.0, 187.0,
     255.0, 263.0, 218.0, 178.0, 351.0, 305.0, 168.0, 334.0, 374.0, 291.0, 385.0, 301.0, 267.0, 265.0, 266.0,
     157.0, 254.0, 234.0, 157.0, 239.0, 337.0, 269.0, 262.0, 242.0, 209.0, 191.0, 186.0, 324.0, 302.0, 164.0,
     283.0, 257.0, 249.0, 196.0, 242.0])

hidden_6 = np.asarray(
    [197.0, 181.0, 266.0, 300.0, 350.0, 369.0, 406.0, 307.0, 253.0, 338.0, 251.0, 229.0, 368.0, 367.0, 328.0, 389.0,
     355.0, 262.0, 268.0, 212.0, 290.0, 235.0, 131.0, 292.0, 367.0, 219.0, 263.0, 182.0, 313.0, 258.0, 272.0, 214.0,
     322.0, 437.0, 193.0, 123.0, 294.0, 391.0, 427.0, 108.0, 297.0, 344.0, 234.0, 371.0, 193.0, 207.0, 369.0, 203.0,
     343.0, 211.0])

hidden_8 = np.asarray(
    [350.0, 291.0, 198.0, 311.0, 197.0, 342.0, 182.0, 314.0, 152.0, 233.0, 321.0, 261.0, 360.0, 433.0, 299.0, 279.0,
     277.0, 341.0, 327.0, 380.0, 183.0, 284.0, 237.0, 346.0, 344.0, 239.0, 170.0, 244.0, 119.0, 328.0, 134.0, 258.0,
     344.0, 317.0, 219.0, 276.0, 273.0, 340.0, 263.0, 246.0, 263.0, 383.0, 286.0, 342.0, 323.0, 282.0, 289.0, 270.0,
     333.0, 206.0])

"""hidden_10 = np.asarray(
    [190.0, 206.0, 260.0, 196.0, 238.0, 282.0, 254.0, 224.0, 268.0, 174.0, 216.0, 222.0, 268.0, 289.0, 259.0, 180.0,
     208.0, 267.0, 257.0, 214.0, 235.0, 198.0, 230.0, 176.0, 221.0, 253.0, 251.0, 219.0, 165.0, 186.0, 233.0, 219.0,
     241.0, 223.0, 180.0, 253.0, 224.0, 225.0, 219.0, 220.0, 206.0, 286.0, 273.0, 245.0, 239.0, 210.0, 226.0, 228.0,
     228.0, 279.0])"""

hidden_10 = np.asarray(
    [295.0, 334.0, 328.0, 235.0, 251.0, 257.0, 276.0, 193.0, 206.0, 249.0, 346.0, 220.0, 294.0, 325.0, 236.0, 359.0,
     329.0, 284.0, 198.0, 231.0, 221.0, 223.0, 222.0, 261.0, 223.0, 336.0, 366.0, 315.0, 281.0, 206.0, 166.0, 295.0,
     200.0, 381.0, 345.0, 275.0, 146.0, 192.0, 161.0, 367.0, 202.0, 215.0, 149.0, 199.0, 257.0, 420.0, 365.0, 212.0,
     276.0, 168.0, 354.0, 228.0, 187.0, 197.0, 376.0, 241.0, 252.0, 337.0, 270.0, 317.0, 315.0, 365.0, 299.0, 309.0,
     288.0, 253.0, 169.0, 383.0, 181.0, 315.0, 313.0, 227.0, 416.0, 285.0, 277.0, 225.0, 239.0, 205.0, 464.0, 238.0,
     113.0, 239.0, 256.0, 365.0, 238.0, 250.0, 132.0, 240.0, 272.0, 306.0, 257.0, 249.0, 261.0, 343.0, 255.0, 186.0,
     288.0, 286.0, 359.0, 158.0])

hidden_15 = np.asarray(
    [206.0, 209.0, 210.0, 214.0, 216.0, 252.0, 212.0, 196.0, 210.0, 249.0, 231.0, 226.0, 203.0, 186.0, 188.0, 254.0,
     183.0, 196.0, 269.0, 181.0, 282.0, 207.0, 208.0, 206.0, 251.0, 262.0, 177.0, 242.0, 253.0, 221.0, 199.0, 200.0,
     222.0, 196.0, 253.0, 204.0, 243.0, 201.0, 172.0, 197.0, 224.0, 177.0, 250.0, 262.0, 203.0, 165.0, 190.0, 248.0,
     267.0, 179.0])

hidden_20 = np.asarray(
    [257.0, 109.0, 252.0, 206.0, 322.0, 349.0, 406.0, 164.0, 339.0, 277.0, 393.0, 351.0, 339.0, 184.0, 230.0, 168.0,
     294.0, 196.0, 257.0, 207.0, 126.0, 293.0, 340.0, 354.0, 330.0, 295.0, 165.0, 408.0, 258.0, 224.0, 255.0, 289.0,
     152.0, 315.0, 338.0, 201.0, 181.0, 356.0, 339.0, 261.0, 220.0, 192.0, 185.0, 393.0, 319.0, 275.0, 311.0, 300.0,
     285.0, 411.0])

data = [hidden_4, hidden_8, hidden_10, hidden_20]

box = plt.boxplot(data, patch_artist=True)

colors = ['steelblue', 'seagreen', 'khaki', 'indianred']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# plt.boxplot()
plt.xticks([1, 2, 3, 4, ], [4, 8, 10, 20])
plt.xlabel('number of hidden nodes')
plt.ylabel('Fitness')
plt.title(label='Fitness vs # hidden nodes',
          fontweight=10,
          pad='2.0')

plt.show()