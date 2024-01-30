"""_summary_"""

from src.common import CircuitJob

from .types import Schedule


def initialize_population(
    circuits: list[CircuitJob], accelerator_capacities: dict[str, int]
) -> list[Schedule]:
    """
    TODO: use bin packing with the following parameters:
    - greedy cutting
    - even cutting
    - "informed cuts"""
    return []


def _greedy_partitioning(
    circuits: list[CircuitJob], accelerator_capacities: dict[str, int]
) -> list[Schedule]:
    return []


def _even_partitioning(
    circuits: list[CircuitJob], accelerator_capacities: dict[str, int]
) -> list[Schedule]:
    return []


def _informed_partitioning(
    circuits: list[CircuitJob], accelerator_capacities: dict[str, int]
) -> list[Schedule]:
    return []
