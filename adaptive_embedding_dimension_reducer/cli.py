"""CLI for Dimension Reducer."""
import argparse
from .core import DimensionReducer

def main():
    parser = argparse.ArgumentParser(description="Adaptive Embedding Dimension Reducer CLI")
    parser.add_argument("--demo", action="store_true", help="Run dimension reduction demo")
    args = parser.parse_args()

    v = [0.1 * i for i in range(1, 1001)]
    red = DimensionReducer.reduce_vector(v, target_dim=64)
    print(f"Original Dim: {red.original_dim} -> Reduced Dim: {red.reduced_dim}")
    print("Sample values:", red.values[:5])

if __name__ == "__main__":
    main()
