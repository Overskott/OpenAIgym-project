import numpy as np

from evolution import *
from genotypes import *
from utils.utils import *




nn = NeuralNetwork('test')

print(nn)

A = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(array_to_input_string(A))
print(array_to_input_string(nn.input_weights))

iw = np.asarray([[0.4116453678139136,-0.5653818689666643,-0.0600552219794932,-0.16313731424026623],
[0.6288816976217166,-1.065524982457163,0.3237210448181329,-1.9359379016732117],
[-0.695072169973024,-0.43120222591003277,-0.19769098560641285,-3.0828840362773673],
[-1.2233440041667734,-1.4498600668039356,-0.362708289158867,-1.6733192825869532]])

test_nn = NeuralNetwork('test', input_weights=iw)

print(array_to_input_string(test_nn.hidden_layer_bias))

print(nn)