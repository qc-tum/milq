"""Full integration test.
Running one circuit on multiple devices,
and reconstructing the results.

"""
from pytest import approx

from src.circuits import create_quantum_only_ghz
from src.common import IBMQBackend
from src.provider import Accelerator, AcceleratorGroup
from src.tools import cut_circuit, optimize_circuit_offline, reconstruct_expvals


def test_tools() -> None:
    """_summary_"""
    backend_belem = IBMQBackend.BELEM
    accelerator_belem = Accelerator(backend_belem)
    backend_quito = IBMQBackend.QUITO
    accelerator_quito = Accelerator(backend_quito)
    accelerator = AcceleratorGroup([accelerator_belem, accelerator_quito])

    circuit = create_quantum_only_ghz(7)
    circuit = optimize_circuit_offline(circuit, backend_belem)
    experiments, uuid = cut_circuit(circuit, [3,4])
    experiments = accelerator.run_experiments(experiments)

    exp_vals = reconstruct_expvals(list(filter(lambda x: x.uuid == uuid, experiments)))
    assert len(exp_vals) == 1
    assert exp_vals[0] == approx(0, abs=0.05)
