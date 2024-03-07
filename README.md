# MILQ

Greedy circuit cutting and a minimal scheduler based on a Mixed Integer Linear Programming problem.

## Installation 

The project is managed with [pdm](https://pdm-project.org/latest/).
Simply use `pdm install` to install all necessary dependencies.
We recommend using a virtual environment.

### Environment variables

This project uses the [Azure Quantum Resource Estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation) for processing time estimation. This requires the following environment variables to be set:
- `AZURE_QUANTUM_WORKSPACE_RESOURCE_ID`
- `AZURE_QUANTUM_WORKSPACE_LOCATION`

In order to not manually authenticate with the system browser you additionally need:
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

## Reproducing results

The `data` module holds the experimental data and setup.
`milp.py` is a sample using synthetic data. 
The benchmark results can be reproduced with `python run_experiments.py`

## Integration

**MILQ** is developed test driven, to see its functionality refer to the test directory.
We use `pytest` as a testing environment.

## References

MILQ implements the methods described in the following paper.

[[1]](https://arxiv.org/abs/2311.17490)
P. Seitz and M. Geiger and C. B. Mendl. Multithreaded parallelism for heterogeneous clusters of QPUs.
