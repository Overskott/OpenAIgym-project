from unittest import TestCase
from unittest import mock
import numpy as np

class Test(TestCase):
    def test_crossover(self):
        import src.evolution as evolution
        bit1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        bit2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        bit3 = np.array([1, 0, 1, 0, 1, 0, 1, 0])


        self.assertTrue(np.array_equal(evolution.crossover(bit1, bit2), bit3))
        assert evolution.crossover(bit1, bit2)
