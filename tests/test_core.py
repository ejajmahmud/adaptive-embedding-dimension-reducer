"""Tests for DimensionReducer."""
import unittest
import math
from adaptive_embedding_dimension_reducer.core import DimensionReducer

class TestDimensionReducer(unittest.TestCase):
    def test_reduction_and_normalization(self):
        raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        red = DimensionReducer.reduce_vector(raw, target_dim=3)
        self.assertEqual(len(red.values), 3)
        norm = math.sqrt(sum(x * x for x in red.values))
        self.assertAlmostEqual(norm, 1.0, places=3)
