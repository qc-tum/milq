# MILQ

A project implementing a minimal scheduler, which is aware of circuit cutting.

## Installation 

The project is managed with [pdm](https://pdm-project.org/latest/).
Simply use `pdm install` to install all necessary dependencies.
We recommend using a virtual environment.

## Reproducing results

The `data` module holds the experimental data and setup.
`milp.py` is a sample using synthetic data. 
The benchmark results can be reproduced with `python run_experiments.py`

## Integration

**MILQ** is developed test driven, to see its functionality refer to the test directory.
We use `pytets` as a testing environment.
