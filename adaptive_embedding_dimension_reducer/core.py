"""Core Dimension Reducer implementation."""
import math
from typing import List

class ReducedVector:
    def __init__(self, values: List[float], original_dim: int, reduced_dim: int):
        self.values = values
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim

class DimensionReducer:
    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    @classmethod
    def reduce_vector(cls, full_vector: List[float], target_dim: int = 128) -> ReducedVector:
        orig_dim = len(full_vector)
        target = min(orig_dim, target_dim)
        sliced = full_vector[:target]
        normalized = cls._l2_normalize(sliced)
        return ReducedVector(normalized, orig_dim, target)

    @classmethod
    def cosine_similarity(cls, vec_a: List[float], vec_b: List[float]) -> float:
        # Assuming normalized vectors, cosine = dot product
        length = min(len(vec_a), len(vec_b))
        return sum(vec_a[i] * vec_b[i] for i in range(length))
