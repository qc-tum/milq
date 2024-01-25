"""Runs the example problem."""
from data.example import example_problem

# Meta Variables
BIG_M = 1000
TIMESTEPS = 2**6

if __name__ == "__main__":
    example_problem(BIG_M, TIMESTEPS,"./data/results/scheduling")
