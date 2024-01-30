"""_summary_"""
from src.scheduling.heuristics import generate_heuristic_info_schedule

from src.circuits import create_quantum_only_ghz


def test_generate_heuristic_info_schedule():
    """Test for generate_heuristic_info_schedule."""
    circuits = [create_quantum_only_ghz(qubits) for qubits in range(2, 6)]
    accelerators = {"a": 5, "b": 5}
    assert len(generate_heuristic_info_schedule(circuits, accelerators)) == 0
