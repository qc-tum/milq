"""_summary_"""

from src.scheduling.heuristics.search import _combine_solutions
from src.scheduling.common.types import Bucket, Machine, Schedule

from src.common import job_from_circuit
from tests.helpers import create_quantum_only_ghz


def test_combine_solutions() -> None:
    """Test set generation in combine solutions"""
    circuits = [job_from_circuit(create_quantum_only_ghz(q)) for q in range(2, 7)]

    buckets = [
        Bucket(circuits[:2]),
        Bucket(circuits[2:4]),
        Bucket([circuits[1], circuits[0]]),
        Bucket(circuits[2:5]),
    ]
    machines = [
        Machine(5, "1", buckets[:2], 0.0),
        Machine(5, "2", buckets[2:4], 0.0),
        Machine(5, "3", buckets[:2], 0.0),
        Machine(5, "3", buckets[:2], 1.0),
        Machine(5, "3", [buckets[2], buckets[1]], 0.0),
    ]

    population = [Schedule(machines[:2], 0.0), Schedule(machines[1:3], 0.0)]
    new_population = [
        Schedule(machines[:2], 1.0),
        Schedule(machines[1::2], 0.0),
        Schedule([machines[1], machines[4]], 0.0),
    ]
    combined = _combine_solutions(population, new_population)
    assert len(combined) == 2
