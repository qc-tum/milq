"""_summary"""
from pytest import approx

from src.circuits import create_quantum_only_ghz
from src.common import IBMQBackend
from src.provider import Accelerator
from src.tools import cut_circuit, optimize_circuit_offline, reconstruct_expvals


def test_reconstrcut_expvals() -> None:
    """_summary_"""
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_quantum_only_ghz(7)
    circuit = optimize_circuit_offline(circuit, backend)
    experiments, uuid = cut_circuit(circuit, [3, 4])
    for experiment in experiments:
        experiment.result_counts = [
            accelerator.run_and_get_counts(circuit) for circuit in experiment.circuits
        ]
    exp_vals = reconstruct_expvals(list(filter(lambda x: x.uuid == uuid, experiments)))
    assert len(exp_vals) == 1
    assert exp_vals[0] == approx(0, abs=0.05)
